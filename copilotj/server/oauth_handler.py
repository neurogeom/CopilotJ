# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Dict

import aiohttp.web as web

from copilotj.util.openai_auth import OpenAIAuthFlow, load_saved_token, get_saved_token_info, remove_saved_token

__all__ = ["OAuthHandler"]

_log = logging.getLogger("copilotj.server.oauth")


class OAuthHandler:
    """Handles OAuth authentication API endpoints."""

    def __init__(self):
        self._auth_flow = OpenAIAuthFlow()
        self._login_tasks: Dict[str, asyncio.Task] = {}

    async def status(self, request: web.Request) -> web.Response:
        """Check authentication status.

        Returns:
            {
                "authenticated": bool,
                "token_info": {
                    "api_key": str (masked),
                    "created_at": str,
                    "expires_at": str | None
                } | None
            }
        """
        try:
            token = load_saved_token()
            if token:
                info = get_saved_token_info()
                # Mask the API key
                if info and "api_key" in info:
                    masked_key = info["api_key"][:8] + "..." + info["api_key"][-4:] if len(info["api_key"]) > 12 else "***"
                    info["api_key"] = masked_key

                return web.json_response({
                    "authenticated": True,
                    "token_info": info
                })
            else:
                return web.json_response({
                    "authenticated": False,
                    "token_info": None
                })
        except Exception as e:
            _log.error(f"Error checking auth status: {e}", exc_info=True)
            return web.json_response({
                "authenticated": False,
                "token_info": None,
                "error": str(e)
            }, status=500)

    async def login(self, request: web.Request) -> web.Response:
        """Initiate OAuth login flow.

        Request body (optional):
            {
                "update_env": bool  # Whether to update .env.local file
            }

        Returns:
            {
                "success": bool,
                "message": str,
                "token": str | None
            }
        """
        try:
            # Parse optional body
            data = await request.json() if request.can_read_body and request.content_length else {}
            update_env = data.get("update_env", False)

            _log.info(f"Starting OAuth login flow (update_env={update_env})")

            # Run authentication
            token = await self._auth_flow.authenticate(update_env=update_env)

            if token:
                _log.info("OAuth login successful")
                # Mask token for response
                masked_token = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
                return web.json_response({
                    "success": True,
                    "message": "Authentication successful",
                    "token": masked_token
                })
            else:
                _log.warning("OAuth login failed: No token received")
                return web.json_response({
                    "success": False,
                    "message": "Authentication failed: No token received",
                    "token": None
                }, status=400)

        except Exception as e:
            _log.error(f"OAuth login error: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "message": f"Authentication error: {str(e)}",
                "token": None
            }, status=500)

    async def logout(self, request: web.Request) -> web.Response:
        """Logout and remove saved credentials.

        Returns:
            {
                "success": bool,
                "message": str
            }
        """
        try:
            removed = remove_saved_token()
            if removed:
                _log.info("User logged out successfully")
                return web.json_response({
                    "success": True,
                    "message": "Logged out successfully"
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": "No active session found"
                }, status=404)
        except Exception as e:
            _log.error(f"Logout error: {e}", exc_info=True)
            return web.json_response({
                "success": False,
                "message": f"Logout error: {str(e)}"
            }, status=500)

    async def get_token(self, request: web.Request) -> web.Response:
        """Get the saved API token.

        Returns:
            {
                "token": str | None,
                "message": str
            }
        """
        try:
            token = load_saved_token()
            if token:
                # For security, we return masked token unless explicitly requested
                query = request.rel_url.query
                unmask = query.get("unmask", "false").lower() == "true"

                if unmask:
                    return web.json_response({
                        "token": token,
                        "message": "Token retrieved"
                    })
                else:
                    masked_token = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
                    return web.json_response({
                        "token": masked_token,
                        "message": "Token retrieved (masked)"
                    })
            else:
                return web.json_response({
                    "token": None,
                    "message": "No saved token found"
                }, status=404)
        except Exception as e:
            _log.error(f"Error retrieving token: {e}", exc_info=True)
            return web.json_response({
                "token": None,
                "message": f"Error: {str(e)}"
            }, status=500)
