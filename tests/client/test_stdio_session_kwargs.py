from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


@pytest.fixture
def sleepy_server(tmp_path: Path) -> Path:
    script = tmp_path / "sleepy_server.py"
    script.write_text(
        "import asyncio\n"
        "from fastmcp import FastMCP\n"
        "server = FastMCP('sleepy')\n"
        "@server.tool\n"
        "async def sleep(seconds: float) -> str:\n"
        "    await asyncio.sleep(seconds)\n"
        "    return 'done'\n"
        "if __name__ == '__main__':\n"
        "    server.run()\n",
        encoding="utf-8",
    )
    return script


@pytest.mark.timeout(20)
@pytest.mark.parametrize("first_timeout, second_timeout", [(1, 3), (3, 1), (3, 3)])
async def test_second_client_uses_its_own_timeout(
    sleepy_server: Path, first_timeout: float, second_timeout: float
) -> None:
    transport = PythonStdioTransport(sleepy_server)
    try:
        async with Client(transport, timeout=first_timeout) as first:
            await first.list_tools()
            first_session = first.session
        async with Client(transport, timeout=second_timeout) as second:
            assert second.session is not first_session
            if second_timeout == 1:
                with pytest.raises(MCPError, match="timed out"):
                    await second.call_tool("sleep", {"seconds": 2})
            else:
                result = await second.call_tool("sleep", {"seconds": 2})
                assert result.data == "done"
    finally:
        await transport.close()


@pytest.mark.timeout(20)
@pytest.mark.parametrize("second_timeout", [1, 3])
async def test_different_client_settings_do_not_replace_an_active_session(
    sleepy_server: Path,
    second_timeout: float,
) -> None:
    transport = PythonStdioTransport(sleepy_server)
    try:
        async with Client(transport, timeout=3) as first:
            with pytest.raises(RuntimeError, match="still using it"):
                async with Client(transport, timeout=second_timeout):
                    pass
            result = await first.call_tool("sleep", {"seconds": 0})
            assert result.data == "done"
    finally:
        await transport.close()


@pytest.mark.timeout(20)
async def test_same_client_reuses_session_and_per_call_timeout_overrides_default(
    sleepy_server: Path,
) -> None:
    transport = PythonStdioTransport(sleepy_server)
    client = Client(transport, timeout=1)
    try:
        async with client:
            session = client.session
        async with client:
            assert client.session is session
            result = await client.call_tool("sleep", {"seconds": 2}, timeout=3)
            assert result.data == "done"
    finally:
        await transport.close()


@pytest.mark.timeout(20)
async def test_matching_session_kwargs_reuse_active_and_idle_connections() -> None:
    script = Path(__file__).parent / "minimal_stdio_server.py"
    transport = PythonStdioTransport(script)
    try:
        async with transport.connect_session(read_timeout_seconds=3) as first:
            await first.initialize()
            async with transport.connect_session(read_timeout_seconds=3) as second:
                assert second is first
        async with transport.connect_session(read_timeout_seconds=3) as third:
            assert third is first
    finally:
        await transport.close()
