"""Trade Journal management for persistent logging and strategy context."""
import datetime
from pathlib import Path


class TradeJournal:
    """Manages a markdown-based trade journal."""

    def __init__(self, path: str = "logs/trades.md") -> None:
        """Initialize the journal."""
        self.path = Path(path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create file with header if it doesn't exist."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = (
                "| Date | Day of Week | Account | Instrument | Side | "
                "Price | Quantity | Value | Currency | Profit | Tax (19%) | Reasoning |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"  # noqa: E501
            )
            self.path.write_text(header, encoding="utf-8")

    def log_trade(
        self,
        account: str,
        instrument: str,
        side: str,
        price: float,
        quantity: float,
        value: float,
        currency: str,
        reasoning: str,
        profit: float = 0.0,
        tax: float = 0.0,
    ) -> None:
        """Append a new trade to the markdown journal."""
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = days[now.weekday()]
        date_str = now.strftime("%Y-%m-%d")

        row = (
            f"| {date_str} | {day_name} | {account} | {instrument} | {side} | "
            f"{price:.8f} | {quantity:.8f} | {value:.2f} | {currency} | "
            f"{profit:.2f} | {tax:.2f} | {reasoning} |\n"
        )

        with self.path.open("a", encoding="utf-8") as f:
            f.write(row)

    def get_last_buy_price(self, ticker: str) -> float | None:
        """Find the most recent buy price for a ticker to calculate profit."""
        if not self.path.exists():
            return None

        try:
            with self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                # Search backwards from the end
                for line in reversed(lines):
                    cols = [c.strip() for s in line.split("|") if (c := s.strip())]
                    # Table index: | Date(0) | Day(1) | Acc(2) | Inst(3) | Side(4) | Price(5) ...
                    # If cols[3] == ticker and cols[4].upper() == 'BUY'
                    if len(cols) > 5 and cols[3] == ticker and cols[4].upper() == "BUY":
                        return float(cols[5])
            return None
        except Exception:  # noqa: BLE001
            return None

    def get_recent_history(self, limit: int = 10, ticker: str | None = None) -> str:
        """Read the last N lines of the journal for strategy context.
        
        If ticker is provided, filters for that specific instrument.
        """
        if not self.path.exists():
            return "No previous trades recorded."

        try:
            with self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                # Skip header (first 2 lines)
                content = lines[2:]

                if ticker:
                    # Filter lines where ticker matches the 4th column (index 3 in split)
                    # Markdown table structure: | Date | Day | Account | Instrument | ...
                    # split('|') results in ['', ' Date ', ' Day ', ' Account ', ' Instrument ', ...]
                    filtered = [
                        line for line in content
                        if len(line.split("|")) > 4 and ticker.lower() in line.split("|")[4].strip().lower()  # noqa: E501, PLR2004
                    ]
                    return "".join(filtered[-limit:])

                return "".join(content[-limit:])
        except Exception:  # noqa: BLE001
            return "Error reading history."
