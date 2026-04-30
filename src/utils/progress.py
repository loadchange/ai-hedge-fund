from datetime import datetime, timezone
import unicodedata
from rich.console import Console
from rich.live import Live
from rich.text import Text
from typing import Dict, Optional, Callable, List
from src.i18n import get_text, translate_agent_name, translate_status

console = Console()


def _display_width(s: str) -> int:
    """Return the terminal display width of a string, accounting for CJK characters."""
    width = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad(s: str, width: int) -> str:
    """Pad string with spaces to reach the target display width."""
    current = _display_width(s)
    if current >= width:
        return s
    return s + " " * (width - current)


class AgentProgress:
    """Manages progress tracking for multiple agents."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, str]] = {}
        self._renderable = Text("")
        # redirect_stdout/stderr=False avoids rich's FileProxy, which leaks a
        # reference past Live.stop() and triggers `ImportError: sys.meta_path
        # is None` on interpreter shutdown when its __del__ tries to flush.
        self.live = Live(
            self._renderable,
            console=console,
            refresh_per_second=4,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.started = False
        self.update_handlers: List[Callable[[str, Optional[str], str], None]] = []

    def register_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """Register a handler to be called when agent status updates."""
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler: Callable[[str, Optional[str], str], None]):
        """Unregister a previously registered handler."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """Start the progress display."""
        if not self.started:
            self.live.start()
            self.started = True

    def stop(self):
        """Stop the progress display."""
        if self.started:
            self.live.stop()
            self.started = False

    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = "", analysis: Optional[str] = None):
        """Update the status of an agent."""
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = {"status": "", "ticker": None}

        if ticker:
            self.agent_status[agent_name]["ticker"] = ticker
        if status:
            self.agent_status[agent_name]["status"] = status
        if analysis:
            self.agent_status[agent_name]["analysis"] = analysis

        timestamp = datetime.now(timezone.utc).isoformat()
        self.agent_status[agent_name]["timestamp"] = timestamp

        for handler in self.update_handlers:
            handler(agent_name, ticker, status, analysis, timestamp)

        self._refresh_display()

    def get_all_status(self):
        """Get the current status of all agents as a dictionary."""
        return {
            agent_name: {
                "ticker": info["ticker"],
                "status": info["status"],
                "display_name": self._get_display_name(agent_name),
            }
            for agent_name, info in self.agent_status.items()
        }

    def _get_display_name(self, agent_name: str) -> str:
        """Convert agent_name to a display-friendly format."""
        name = agent_name.replace("_agent", "").replace("_", " ").title()
        return translate_agent_name(name)

    def _refresh_display(self):
        """Refresh the progress display as aligned text lines."""

        def sort_key(item):
            agent_name = item[0]
            if "risk_management" in agent_name:
                return (2, agent_name)
            elif "portfolio_management" in agent_name:
                return (3, agent_name)
            else:
                return (1, agent_name)

        # Pre-compute column widths
        names = []
        tickers = []
        for agent_name, info in self.agent_status.items():
            names.append(self._get_display_name(agent_name))
            tickers.append(f"[{info['ticker']}]" if info.get("ticker") else "")
        name_width = max((_display_width(n) for n in names), default=16)
        name_width = max(name_width, 16)
        ticker_width = max((_display_width(t) for t in tickers), default=8)
        ticker_width = max(ticker_width, 8)

        output = Text()
        for i, (agent_name, info) in enumerate(
            sorted(self.agent_status.items(), key=sort_key)
        ):
            status = info["status"]
            ticker = info.get("ticker")
            display_name = self._get_display_name(agent_name)
            ticker_str = f"[{ticker}]" if ticker else ""

            if status.lower() == "done":
                symbol = Text(" ✓ ", style="green bold")
                name_text = Text(_pad(display_name, name_width), style="bold")
                ticker_text = Text(_pad(ticker_str, ticker_width), style="cyan")
                status_text = Text(get_text("done"), style="green bold")
            elif status.lower() == "error":
                symbol = Text(" ✗ ", style="red bold")
                name_text = Text(_pad(display_name, name_width), style="bold")
                ticker_text = Text(_pad(ticker_str, ticker_width), style="cyan")
                status_text = Text(get_text("error"), style="red bold")
            else:
                symbol = Text(" ⋯ ", style="yellow")
                name_text = Text(_pad(display_name, name_width), style="bold")
                ticker_text = Text(_pad(ticker_str, ticker_width), style="cyan")
                status_text = Text(translate_status(status), style="yellow")

            output.append_text(symbol)
            output.append_text(name_text)
            output.append_text(ticker_text)
            output.append_text(status_text)
            if i < len(self.agent_status) - 1:
                output.append("\n")

        self._renderable = output
        self.live.update(output)


# Create a global instance
progress = AgentProgress()
