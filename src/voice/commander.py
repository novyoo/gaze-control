# src/voice/commander.py

import os
import re
import time
import logging
import winreg
import webbrowser
import pyautogui
import pyperclip
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    success: bool
    command: str
    action:  str
    text:    str


class VoiceCommander:

    def __init__(self):
        self._last_command_time = 0.0
        self._cooldown          = 0.4
        self._paused            = False
        self._app_cache         = {}
        self._cache_built       = False

    def process(self, text: str) -> Optional[CommandResult]:
        now  = time.time()
        if now - self._last_command_time < self._cooldown:
            return None

        text = text.lower().strip()
        logger.info(f"Processing: '{text}'")

        # ── Meta ──────────────────────────────────────────────────────
        if any(w in text for w in ["stop listening", "pause listening"]):
            self._paused = True
            return CommandResult(True, text, "paused", text)

        if any(w in text for w in ["start listening", "resume listening"]):
            self._paused = False
            return CommandResult(True, text, "resumed", text)

        if self._paused:
            return None

        # ── Type at cursor ────────────────────────────────────────────
        if text.startswith("type "):
            query = text[5:].strip()
            if query:
                pyautogui.click()
                time.sleep(0.15)
                pyperclip.copy(query)
                pyautogui.hotkey("ctrl", "v")
                self._last_command_time = now
                return CommandResult(True, text, f"typed: {query}", text)

        # ── Search ────────────────────────────────────────────────────
        if text.startswith("search "):
            query = text[7:].strip()
            if query:
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                webbrowser.open(url)
                self._last_command_time = now
                return CommandResult(True, text, f"searched: {query}", text)

        # ── Open app ──────────────────────────────────────────────────
        if text.startswith("open "):
            app_name = text[5:].strip()
            return self._open_app(app_name, text, now)

        # ── Screenshot ────────────────────────────────────────────────
        if "screenshot" in text:
            try:
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                os.makedirs(desktop, exist_ok=True)
                path = os.path.join(desktop, f"screenshot_{int(now)}.png")
                pyautogui.screenshot().save(path)
                self._last_command_time = now
                return CommandResult(True, text, "saved to Desktop", text)
            except Exception as e:
                return CommandResult(False, text, f"screenshot failed: {e}", text)

        # ── Volume ────────────────────────────────────────────────────
        if "volume up" in text:
            amount = self._extract_number(text, default=5)
            for _ in range(amount):
                pyautogui.press("volumeup")
            self._last_command_time = now
            return CommandResult(True, text, f"volume up x{amount}", text)

        if "volume down" in text:
            amount = self._extract_number(text, default=5)
            for _ in range(amount):
                pyautogui.press("volumedown")
            self._last_command_time = now
            return CommandResult(True, text, f"volume down x{amount}", text)

        if "mute" in text:
            pyautogui.press("volumemute")
            self._last_command_time = now
            return CommandResult(True, text, "muted", text)

        # ── Scroll ────────────────────────────────────────────────────
        if "scroll up" in text:
            amount = self._extract_number(text, default=5)
            pyautogui.scroll(amount * 100)
            self._last_command_time = now
            return CommandResult(True, text, f"scrolled up x{amount}", text)

        if "scroll down" in text:
            amount = self._extract_number(text, default=5)
            pyautogui.scroll(-amount * 100)
            self._last_command_time = now
            return CommandResult(True, text, f"scrolled down x{amount}", text)

        # ── No match ──────────────────────────────────────────────────
        return CommandResult(False, text, "no match", text)

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ── App finder ────────────────────────────────────────────────────
    def _open_app(self, app_name: str, original_text: str,
                  now: float) -> CommandResult:
        if not self._cache_built:
            self._build_app_cache()

        match = self._find_in_cache(app_name)
        if match:
            try:
                os.startfile(match)
                self._last_command_time = now
                return CommandResult(True, original_text,
                                     f"opened: {app_name}", original_text)
            except Exception as e:
                logger.error(f"startfile failed: {e}")

        try:
            os.system(f'start "" "{app_name}"')
            self._last_command_time = now
            return CommandResult(True, original_text,
                                 f"opened: {app_name}", original_text)
        except Exception:
            pass

        path = self._search_start_menu(app_name)
        if path:
            try:
                os.startfile(path)
                self._app_cache[app_name] = path
                self._last_command_time = now
                return CommandResult(True, original_text,
                                     f"opened: {app_name}", original_text)
            except Exception as e:
                logger.error(f"Shortcut launch failed: {e}")

        return CommandResult(False, original_text,
                             f"could not find: {app_name}", original_text)

    def _build_app_cache(self):
        logger.info("Building app cache — scanning device...")

        start_menu_dirs = [
            os.path.join(os.environ.get("APPDATA", ""),
                         r"Microsoft\Windows\Start Menu\Programs"),
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
        ]
        for directory in start_menu_dirs:
            if not os.path.exists(directory):
                continue
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith(".lnk"):
                        name = f[:-4].lower()
                        self._app_cache[name] = os.path.join(root, f)

        reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
            try:
                key = winreg.OpenKey(hive, reg_path)
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        subkey      = winreg.OpenKey(key, subkey_name)
                        path, _     = winreg.QueryValueEx(subkey, "")
                        name        = subkey_name.replace(".exe", "").lower()
                        self._app_cache[name] = path
                    except Exception:
                        continue
            except Exception:
                continue

        search_dirs = [
            r"C:\Program Files",
            r"C:\Program Files (x86)",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        ]
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
            for root, dirs, files in os.walk(directory):
                depth = root.replace(directory, "").count(os.sep)
                if depth > 2:
                    dirs.clear()
                    continue
                for f in files:
                    if f.endswith(".exe"):
                        name = f[:-4].lower()
                        if name not in self._app_cache:
                            self._app_cache[name] = os.path.join(root, f)

        self._cache_built = True
        logger.info(f"App cache built: {len(self._app_cache)} apps found")

    def _find_in_cache(self, app_name: str) -> Optional[str]:
        app_name = app_name.lower().strip()

        if app_name in self._app_cache:
            return self._app_cache[app_name]

        for key, path in self._app_cache.items():
            if app_name in key:
                return path

        for key, path in self._app_cache.items():
            if key in app_name:
                return path

        return None

    def _search_start_menu(self, app_name: str) -> Optional[str]:
        start_menu_dirs = [
            os.path.join(os.environ.get("APPDATA", ""),
                         r"Microsoft\Windows\Start Menu\Programs"),
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
        ]
        app_name = app_name.lower()
        for directory in start_menu_dirs:
            if not os.path.exists(directory):
                continue
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if app_name in f.lower() and f.endswith(".lnk"):
                        return os.path.join(root, f)
        return None

    def _extract_number(self, text: str, default: int) -> int:
        numbers = re.findall(r'\d+', text)
        if numbers:
            return min(int(numbers[-1]), 50)
        return default