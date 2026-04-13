# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import aiohttp.web as web
import pydantic

from copilotj.util import Base64ImageTruncator, IndentedRawJson

__all__ = ["BridgeRequest", "BridgeResponse", "Bridge", "DEV_CLIENT_ID"]

_log = logging.getLogger("copilotj.server.bridge")
TEXT_MESSAGE_QUEUE_MAX_SIZE = 128
DEFAULT_TIMEOUT = 4
DEV_CLIENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class _BridgeEvent(pydantic.BaseModel):
    id: uuid.UUID
    event_id: uuid.UUID
    event: str
    data: Any | None
    err: str | None


class BridgeRequest(pydantic.BaseModel):
    client_id: uuid.UUID
    event: str
    data: Any | None = None
    timeout: float | None = None  # seconds


class BridgeResponse(pydantic.BaseModel):
    data: Any | None = None
    err: str | None = None


class _IdChangedEventData(pydantic.BaseModel):
    id: uuid.UUID


class _Client:
    def __init__(self, bridge: "Bridge") -> None:
        super().__init__()
        self.id = uuid.uuid4()
        self._bridge = bridge
        self._ws = web.WebSocketResponse()
        self._text_message_queue = asyncio.Queue[str](maxsize=TEXT_MESSAGE_QUEUE_MAX_SIZE)
        self._registered_events = dict[uuid.UUID, asyncio.Future[BridgeResponse]]()
        self._events_lock = asyncio.Lock()

    async def run(self, request: web.Request) -> web.StreamResponse:
        await self._ws.prepare(request)

        # Create a task to listen for messages from the queue and send them to the websocket
        async def websocket_sender():
            try:
                while not self._ws.closed:
                    message = await self._text_message_queue.get()
                    if self._ws.closed:  # FIXME: put the message back in the queue?
                        break

                    try:
                        await self._ws.send_str(message)
                    except Exception as e:
                        _log.exception(f"Failed to send message to WebSocket: {e}")
                    finally:
                        self._text_message_queue.task_done()

            except asyncio.CancelledError:
                _log.info("WebSocket sender task cancelled")

        sender_task = asyncio.ensure_future(websocket_sender())

        try:
            await self.on_message(self._ws)

        except Exception as e:
            _log.exception(e)

        finally:
            _log.info("Bridge WebSocket connection closed")
            await self._ws.close()

            # Cancel the sender task
            if not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    _log.info("Sender task cancelled")

        return self._ws

    async def on_message(self, ws: web.WebSocketResponse) -> None:
        async for msg in ws:
            match msg.type:
                case web.WSMsgType.TEXT:
                    _log.debug("Received: %s", IndentedRawJson(Base64ImageTruncator(msg.data)))
                    await self._on_message_text(msg)

                case web.WSMsgType.PING:
                    await ws.pong()

                case web.WSMsgType.PONG:
                    _log.debug("Received pong")

                case web.WSMsgType.CLOSE:
                    _log.info("Received CLOSE")
                    break

                case web.WSMsgType.ERROR:
                    _log.info("ws connection closed with exception %s", ws.exception())
                    break

                case _:
                    _log.error("Received non-recognized message: %s", msg.type)
                    break

    async def send_event(self, event: str, data: Any, *, timeout: float = DEFAULT_TIMEOUT) -> BridgeResponse:
        # Create a future to track the response
        future = asyncio.get_event_loop().create_future()
        event_id = uuid.uuid4()
        async with self._events_lock:
            self._registered_events[event_id] = future

        try:
            payload = _BridgeEvent(id=uuid.uuid4(), event_id=event_id, event=event, data=data, err=None)
            await self._text_message_queue.put(payload.model_dump_json())
            _log.debug("Sent event: %s (id: %s) with %s", event, event_id, data)

            # Wait for the response with timeout
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                _log.warning("Timeout waiting for response to event: %s (id: %s)", event, event_id)
                raise Exception(f"Timeout waiting for response to event: {event} (id: {event_id})")

        finally:
            # Clean up the registered event
            async with self._events_lock:
                self._registered_events.pop(event_id, None)

    async def close(self) -> None:
        await self._ws.close(code=aiohttp.WSCloseCode.GOING_AWAY, message=b"Server shutdown")

    async def _on_message_text(self, msg: aiohttp.WSMessage) -> None:
        try:
            # TODO: pref: streaming and partial validation
            response = _BridgeEvent.model_validate_json(msg.data)

        except Exception as e:
            print(e)
            _log.error(f"Failed to handle message: {e}")
            return

        # check if the event is registered
        event_id = response.event_id
        async with self._events_lock:
            if event_id not in self._registered_events:
                future = None
            else:
                future = self._registered_events[event_id]

        if future is None:
            return await self._on_event(response)

        if future.done():
            _log.warning("Received event ID: %s (event: %s), but future is done", event_id, response.event)
            return

        _log.debug("Received unregistered event ID: %s (event: %s)", event_id, response.event)
        resp = BridgeResponse(data=response.data, err=response.err)
        future.set_result(resp)

    async def _on_event(self, response: _BridgeEvent) -> None:
        match response.event:
            case "query_id":
                data = _IdChangedEventData(id=self.id)
                payload = _BridgeEvent(
                    id=uuid.uuid4(), event_id=response.event_id, event="id_changed", data=data, err=None
                )
                await self._text_message_queue.put(payload.model_dump_json())

            case "negotiate_id":
                event_data = _IdChangedEventData.model_validate(response.data)
                result = self._bridge._negotiate_id(self, event_data.id)
                if result is not None:
                    err = result
                else:
                    self.id = event_data.id
                    err = None

                data = _IdChangedEventData(id=self.id)  # always send the current id
                payload = _BridgeEvent(
                    id=uuid.uuid4(), event_id=response.event_id, event="id_changed", data=data, err=err
                )

            case _:
                _log.warning("Received unrecognized event: %s (id: %s)", response.event, response.id)
                return

        await self._text_message_queue.put(payload.model_dump_json())


class Bridge:
    def __init__(self) -> None:
        super().__init__()
        self._clients = dict[uuid.UUID, _Client]()
        self._close_event: asyncio.Event | None = None  # TODO
        self._used_client_ids = dict[uuid.UUID, datetime]()

    async def on_plugin_connect(self, request: web.Request) -> web.StreamResponse:
        client = _Client(self)
        self._clients[client.id] = client
        self._used_client_ids[client.id] = datetime.now()
        _log.info("Client WebSocket connection established: %s", client.id)

        ws = await client.run(request)
        self._clients.pop(client.id, None)
        return ws

    async def on_forward_event(self, request: web.Request) -> web.Response:
        """Handle forwarding events to ImageJ."""
        try:
            # Parse request body
            body = await request.read()
            req = BridgeRequest.model_validate_json(body)
            resp = await self.send_event(req)
            state = 200 if resp.err is None else 500

        except Exception as e:
            _log.error(f"Error forwarding event: {e}")
            resp = BridgeResponse(data=None, err=str(e))
            state = 500

        return web.Response(status=state, text=resp.model_dump_json(), content_type="application/json")

    async def send_event(self, req: BridgeRequest) -> BridgeResponse:
        client_id = req.client_id
        if client_id == DEV_CLIENT_ID and len(self._clients) > 0:
            client_id = next(iter(self._clients.keys()))  # get the first client

        client = self._clients.get(client_id)
        if client is None:
            _log.error(f"Client not found: {req.client_id}")
            return BridgeResponse(err="Client not found")

        # Forward the event
        timeout = req.timeout or DEFAULT_TIMEOUT
        return await client.send_event(req.event, req.data, timeout=timeout)

    async def close(self) -> None:
        for _, c in self._clients.items():
            await c.close()

        self._clients.clear()

    def _negotiate_id(self, client: _Client, new_id: uuid.UUID) -> str | None:
        if new_id == client.id:  # not changed
            return None

        if (
            new_id == DEV_CLIENT_ID  # dev client id
            or new_id not in self._used_client_ids  # not used
            or new_id in self._clients  # already used
            or self._clients[client.id] != client  # error from the client
        ):
            return "Client ID already used"

        # only allow changing the id if it was used in the last 24 hours
        if self._used_client_ids[new_id] - datetime.now() > timedelta(days=1):
            del self._used_client_ids[new_id]  # remove the old id
            return "Client Id already used"

        self._used_client_ids[new_id] = datetime.now()  # update the used id
        self._clients[new_id] = client  # update the client id
        del self._clients[client.id]  # remove the old client id
        return None
