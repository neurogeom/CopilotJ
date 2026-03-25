# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI OAuth 2.0 Authentication Flow with PKCE
Based on: https://github.com/pedrobrantes/codex-get-auth-conf

This module implements the OAuth 2.0 Authorization Code flow with PKCE
(Proof Key for Code Exchange) to obtain OpenAI API keys programmatically.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

__all__ = ["OpenAIAuthFlow", "get_openai_token", "load_saved_token", "get_saved_token_info", "remove_saved_token"]


# OpenAI OAuth Configuration
ISSUER = "https://auth.openai.com"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # Public Client ID for Codex CLI
REDIRECT_PORT = 1455
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/auth/callback"


class OpenAIAuthFlow:
    """Handles OpenAI OAuth 2.0 authentication with PKCE."""

    def __init__(
        self,
        issuer: str = ISSUER,
        client_id: str = CLIENT_ID,
        redirect_port: int = REDIRECT_PORT,
    ):
        self.issuer = issuer
        self.client_id = client_id
        self.redirect_port = redirect_port
        self.redirect_uri = f"http://localhost:{redirect_port}/auth/callback"

        # PKCE parameters
        self.code_verifier: str = ""
        self.code_challenge: str = ""
        self.state: str = ""

        # Results
        self.access_token: str = ""
        self.tokens: dict[str, Any] = {}

        # Server
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # Event to signal completion
        self._auth_complete = asyncio.Event()
        self._error: Exception | None = None

    def _generate_pkce_codes(self) -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge."""
        # Generate random code_verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode('utf-8').rstrip('=')

        # Create code_challenge using SHA256
        challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')

        return code_verifier, code_challenge

    async def _get_oidc_configuration(self) -> dict[str, str]:
        """Fetch OpenAI's OIDC configuration."""
        discovery_url = f"{self.issuer}/v2.0/.well-known/openid-configuration"

        async with aiohttp.ClientSession() as session:
            async with session.get(discovery_url) as resp:
                if not resp.ok:
                    raise Exception("Failed to fetch OIDC configuration")
                config = await resp.json()

                # Override endpoints
                config['token_endpoint'] = f"{self.issuer}/oauth/token"
                config['authorization_endpoint'] = f"{self.issuer}/oauth/authorize"

                return config

    def _build_authorization_url(self, oidc_config: dict[str, str]) -> str:
        """Build the authorization URL."""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid profile email offline_access",
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "state": self.state,
        }

        auth_url = f"{oidc_config['authorization_endpoint']}?{urlencode(params)}"
        return auth_url

    async def _exchange_code_for_tokens(
        self,
        code: str,
        token_endpoint: str
    ) -> dict[str, Any]:
        """Exchange authorization code for tokens."""
        params = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self.code_verifier,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_endpoint,
                data=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise Exception(f"Failed to exchange code for tokens: {error_text}")
                return await resp.json()

    async def _exchange_token_for_api_key(
        self,
        id_token: str,
        token_endpoint: str
    ) -> dict[str, Any]:
        """Exchange ID token for OpenAI API key."""
        random_id = secrets.token_hex(6)
        date_str = datetime.now().strftime("%Y-%m-%d")

        params = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": self.client_id,
            "requested_token": "openai-api-key",
            "subject_token": id_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
            "name": f"ChatImageJ [auto-generated] ({date_str}) [{random_id}]",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                token_endpoint,
                data=params,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise Exception(f"Failed to create API key: {error_text}")
                return await resp.json()

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handle OAuth callback."""
        try:
            # Validate state
            query_params = request.rel_url.query
            received_state = query_params.get("state")

            if not received_state or received_state != self.state:
                raise Exception("Invalid state parameter")

            # Get authorization code
            code = query_params.get("code")
            if not code:
                raise Exception("Missing authorization code")

            # Get OIDC config
            oidc_config = await self._get_oidc_configuration()
            token_endpoint = oidc_config["token_endpoint"]

            # Exchange code for tokens
            print("Exchanging authorization code for tokens...")
            token_data = await self._exchange_code_for_tokens(code, token_endpoint)

            # Exchange ID token for API key
            print("Creating OpenAI API key...")
            api_key_data = await self._exchange_token_for_api_key(
                token_data["id_token"],
                token_endpoint
            )

            # Save results
            self.access_token = api_key_data["access_token"]
            self.tokens = {
                **token_data,
                "api_key": self.access_token,
            }

            # Save to file
            self._save_auth_file()

            # Signal completion
            self._auth_complete.set()

            # Return success page
            return web.Response(
                text=self._get_success_html(),
                content_type="text/html"
            )

        except Exception as e:
            self._error = e
            self._auth_complete.set()
            return web.Response(
                text=f"<html><body><h1>Authentication Failed</h1><p>{str(e)}</p></body></html>",
                content_type="text/html",
                status=500
            )

    def _save_auth_file(self) -> None:
        """Save authentication data to ~/.chatimej/auth.json."""
        auth_dir = Path.home() / ".chatimej"
        auth_dir.mkdir(exist_ok=True)

        auth_file = auth_dir / "auth.json"
        auth_data = {
            "tokens": self.tokens,
            "last_refresh": datetime.now().isoformat(),
            "OPENAI_API_KEY": self.access_token,
            "created_at": datetime.now().isoformat(),
        }

        # Write with restricted permissions
        auth_file.write_text(json.dumps(auth_data, indent=2))
        auth_file.chmod(0o600)

        print(f"\n✓ Auth data saved to: {auth_file}")

    def _get_success_html(self) -> str:
        """Return success HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>ChatImageJ - Signed In</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }
        .container {
            text-align: center;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .success-icon {
            font-size: 64px;
            color: #04B84C;
            margin-bottom: 20px;
        }
        h1 {
            color: #0D0D0D;
            margin-bottom: 10px;
        }
        p {
            color: #5D5D5D;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="success-icon">✓</div>
        <h1>Successfully Signed In!</h1>
        <p>Your OpenAI API key has been obtained and saved.</p>
        <p>You may now close this window and return to ChatImageJ.</p>
    </div>
</body>
</html>
"""

    async def authenticate(self, update_env: bool = False) -> str:
        """
        Start the OAuth authentication flow.

        Args:
            update_env: Whether to update .env.local file with the API key

        Returns:
            str: The obtained OpenAI API key
        """
        self.update_env = update_env
        print(f"\n{'='*60}")
        print("ChatImageJ - OpenAI Authentication")
        print(f"{'='*60}\n")

        # Generate PKCE codes and state
        self.code_verifier, self.code_challenge = self._generate_pkce_codes()
        self.state = secrets.token_hex(32)

        # Setup routes
        self.app.router.add_get("/auth/callback", self._handle_callback)

        # Start local server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.redirect_port)

        try:
            await self.site.start()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                raise Exception(
                    f"Port {self.redirect_port} is already in use. "
                    "You might already be signing in."
                )
            raise

        print(f"✓ Local callback server started on port {self.redirect_port}")

        # Get OIDC config and build auth URL
        oidc_config = await self._get_oidc_configuration()
        auth_url = self._build_authorization_url(oidc_config)

        # Open browser
        print(f"\n🌐 Opening browser for authentication...")
        print(f"   URL: {auth_url}\n")
        webbrowser.open(auth_url)

        # Wait for callback
        print("⏳ Waiting for you to complete the sign-in in your browser...")
        await self._auth_complete.wait()

        # Cleanup
        await self.runner.cleanup()

        # Check for errors
        if self._error:
            raise self._error

        print(f"\n{'='*60}")
        print("✓ Authentication successful!")
        print(f"{'='*60}\n")
        print(f"API Key: {self.access_token[:20]}...{self.access_token[-10:]}")

        return self.access_token


async def get_openai_token(update_env: bool = False) -> str:
    """
    Authenticate with OpenAI and obtain an API key.

    Args:
        update_env: Whether to update .env.local file with the API key

    Returns:
        str: OpenAI API key
    """
    auth_flow = OpenAIAuthFlow()
    return await auth_flow.authenticate(update_env=update_env)


def load_saved_token() -> str | None:
    """
    Load previously saved OpenAI token from ~/.chatimej/auth.json

    Returns:
        str | None: API key if found, None otherwise
    """
    auth_file = Path.home() / ".chatimej" / "auth.json"

    if not auth_file.exists():
        return None

    try:
        auth_data = json.loads(auth_file.read_text())
        return auth_data.get("OPENAI_API_KEY")
    except Exception as e:
        print(f"Warning: Failed to load saved token: {e}")
        return None


def get_saved_token_info() -> dict[str, Any] | None:
    """
    Get information about the saved token.

    Returns:
        dict with token info or None
    """
    auth_file = Path.home() / ".chatimej" / "auth.json"

    if not auth_file.exists():
        return None

    try:
        auth_data = json.loads(auth_file.read_text())
        api_key = auth_data.get("OPENAI_API_KEY")
        created_at = auth_data.get("created_at")

        if not api_key:
            return None

        return {
            "api_key": api_key,
            "created_at": created_at,
            "expires_at": None  # OpenAI keys don't expire by default
        }
    except Exception as e:
        print(f"Warning: Failed to get token info: {e}")
        return None


def remove_saved_token() -> bool:
    """
    Remove saved authentication credentials.

    Returns:
        bool: True if removed successfully, False if no file existed
    """
    auth_file = Path.home() / ".chatimej" / "auth.json"

    if not auth_file.exists():
        return False

    try:
        auth_file.unlink()
        print(f"✓ Removed authentication file: {auth_file}")
        return True
    except Exception as e:
        print(f"Warning: Failed to remove auth file: {e}")
        return False


if __name__ == "__main__":
    # CLI usage
    async def main():
        try:
            api_key = await get_openai_token()
            print(f"\n✓ Success! Your API key has been saved.")
            print(f"\nTo use in ChatImageJ, add to your .env.local:")
            print(f"COPILOTJ_API_KEY={api_key}")
        except Exception as e:
            print(f"\n✗ Error: {e}")

    asyncio.run(main())
