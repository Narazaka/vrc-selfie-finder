from __future__ import annotations

import csv
import logging
import os
import threading
from pathlib import Path

import flet as ft

from .config import DEFAULT_PHOTO_DIR, Config
from .pipeline import run_pipeline


class VsfGui:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "vrc-selfie-finder"
        self.page.window.width = 1100
        self.page.window.height = 750
        self.page.window.prevent_close = True

        self._running = False
        self._closing = False
        self._result_data: dict[str, list[tuple[str, float]]] = {}  # avatar -> [(path, score)]

        self.page.window.on_event = self._on_window_event

        # --- FilePicker (services に登録) ---
        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)

        # --- 設定コントロール ---
        self.photo_dir_field = ft.TextField(
            label="写真フォルダ",
            value=str(DEFAULT_PHOTO_DIR),
            expand=True,
            read_only=True,
            dense=True,
        )
        self.reference_dir_field = ft.TextField(
            label="リファレンス",
            value="reference_images",
            expand=True,
            read_only=True,
            dense=True,
        )
        self.output_dir_field = ft.TextField(
            label="出力先",
            value="output",
            expand=True,
            read_only=True,
            dense=True,
        )

        self.matcher_radio = ft.RadioGroup(
            ft.Row([
                ft.Radio(value="ccip", label="CCIP"),
                ft.Radio(value="clip", label="CLIP"),
            ]),
            value="ccip",
        )

        self.threshold_slider = ft.Slider(
            min=0.5, max=1.0, value=0.87, divisions=50,
            label="{value}",
            on_change=self._on_threshold_change,
            expand=True,
        )
        self.threshold_text = ft.Text("0.87", size=13, width=40)

        self.crop_dropdown = ft.Dropdown(
            label="切り抜き",
            value="wide",
            options=[
                ft.DropdownOption(key="face", text="face"),
                ft.DropdownOption(key="wide", text="wide"),
                ft.DropdownOption(key="full", text="full"),
            ],
            dense=True,
            width=120,
        )

        self.rotation_checkbox = ft.Checkbox(label="回転検出", value=True)

        self.since_field = ft.TextField(
            label="since", hint_text="YYYY-MM-DD", dense=True, width=130,
        )
        self.until_field = ft.TextField(
            label="until", hint_text="YYYY-MM-DD", dense=True, width=130,
        )

        self.run_button = ft.ElevatedButton(
            "実行",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run,
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
        )

        # --- 進捗 ---
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.progress_text = ft.Text("", size=12)

        # --- ログ ---
        self.log_list = ft.ListView(expand=True, spacing=2, auto_scroll=True)

        # --- ソート ---
        self.sort_dropdown = ft.Dropdown(
            value="name",
            options=[
                ft.DropdownOption(key="name", text="名前順"),
                ft.DropdownOption(key="score", text="スコア順"),
            ],
            dense=True,
            width=110,
            on_select=self._on_sort_change,
        )
        self.sort_order_button = ft.IconButton(
            icon=ft.Icons.ARROW_UPWARD,
            tooltip="昇順",
            on_click=self._on_sort_order_toggle,
            icon_size=18,
        )
        self._sort_ascending = True

        # --- 結果 (動的に差し替える) ---
        self.result_container = ft.Container(
            content=ft.Text("結果がここに表示されます", size=14, color=ft.Colors.GREY),
            expand=True,
            padding=10,
            alignment=ft.Alignment.CENTER,
        )

        # --- レイアウト組み立て ---
        settings_panel = ft.Container(
            content=ft.Column([
                self._dir_row(self.photo_dir_field),
                self._dir_row(self.reference_dir_field),
                self._dir_row(self.output_dir_field),
                ft.Divider(height=8),
                ft.Text("マッチャー", size=12),
                self.matcher_radio,
                ft.Row([ft.Text("閾値", size=12), self.threshold_slider, self.threshold_text]),
                ft.Row([self.crop_dropdown, self.rotation_checkbox]),
                ft.Row([self.since_field, self.until_field]),
                ft.Container(height=8),
                self.run_button,
            ], spacing=6, scroll=ft.ScrollMode.AUTO),
            width=310,
            padding=10,
        )

        bottom_panel = ft.Container(
            content=ft.Column([
                ft.Row([self.progress_bar, self.progress_text], spacing=10),
                ft.Container(content=self.log_list, height=120, border=ft.border.all(1, ft.Colors.OUTLINE)),
            ], spacing=4),
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
        )

        self.page.add(
            ft.Row(
                [settings_panel, ft.VerticalDivider(width=1), self.result_container],
                expand=True,
            ),
            ft.Divider(height=1),
            bottom_panel,
        )

    # --- ヘルパー ---

    async def _on_window_event(self, e: ft.WindowEvent):
        if e.type == ft.WindowEventType.CLOSE:
            self._closing = True
            self.page.window.prevent_close = False
            await self.page.window.destroy()
            os._exit(0)

    def _safe_update(self):
        if not self._closing:
            try:
                self.page.update()
            except Exception:
                self._closing = True

    def _dir_row(self, field: ft.TextField) -> ft.Row:
        async def on_pick(_e):
            result = await self.file_picker.get_directory_path()
            if result:
                field.value = result
                self.page.update()

        return ft.Row([
            field,
            ft.IconButton(
                icon=ft.Icons.FOLDER_OPEN,
                on_click=on_pick,
                tooltip="フォルダを選択",
            ),
        ], spacing=0)

    def _on_threshold_change(self, _e):
        self.threshold_text.value = f"{self.threshold_slider.value:.2f}"
        self.page.update()

    # --- 実行 ---

    def _build_config(self) -> Config:
        return Config(
            photo_dir=Path(self.photo_dir_field.value),
            reference_dir=Path(self.reference_dir_field.value),
            output_dir=Path(self.output_dir_field.value),
            matcher=self.matcher_radio.value,
            similarity_threshold=round(self.threshold_slider.value, 2),
            crop_mode=self.crop_dropdown.value,
            try_rotations=self.rotation_checkbox.value,
            since=self.since_field.value or None,
            until=self.until_field.value or None,
        )

    def _on_run(self, _e):
        if self._running:
            return
        self._running = True
        self.run_button.disabled = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.progress_text.value = ""
        self.log_list.controls.clear()
        self.page.update()

        config = self._build_config()

        last_pct = {"value": -1}

        def _append_log(msg: str):
            self.log_list.controls.append(ft.Text(msg, size=11, selectable=True))
            if len(self.log_list.controls) > 500:
                self.log_list.controls.pop(0)

        def on_log(msg: str):
            if self._closing:
                return
            _append_log(msg)
            self._safe_update()

        def on_progress(label: str, current: int, total: int):
            if self._closing:
                return
            if total > 0:
                pct = int(current / total * 100)
                # 5%刻みでログに表示（大量のログ行を避ける）
                if pct >= last_pct["value"] + 5 or current == total:
                    last_pct["value"] = pct
                    _append_log(f"  {label} {current}/{total} ({pct}%)")
                # プログレスバーも更新を試みる
                self.progress_bar.value = current / total
                self.progress_text.value = f"{label} {current}/{total} ({pct}%)"
            else:
                last_pct["value"] = -1
                self.progress_bar.value = None  # indeterminate
                self.progress_text.value = label
                _append_log(label)
            self._safe_update()

        def run():
            try:
                run_pipeline(config, on_progress=on_progress, on_log=on_log)
                if not self._closing:
                    on_log("[完了] 結果を表示します。")
                    self._load_results(config.output_dir)
            except Exception as exc:
                if not self._closing:
                    import traceback
                    on_log(f"[エラー] {exc}")
                    on_log(traceback.format_exc())
            finally:
                self._running = False
                if not self._closing:
                    self.run_button.disabled = False
                    self.progress_bar.visible = False
                    self._safe_update()

        threading.Thread(target=run, daemon=True).start()

    # --- ソート ---

    def _on_sort_change(self, _e):
        self._rebuild_result_view()
        self.page.update()

    def _on_sort_order_toggle(self, _e):
        self._sort_ascending = not self._sort_ascending
        self.sort_order_button.icon = ft.Icons.ARROW_UPWARD if self._sort_ascending else ft.Icons.ARROW_DOWNWARD
        self.sort_order_button.tooltip = "昇順" if self._sort_ascending else "降順"
        self._rebuild_result_view()
        self.page.update()

    def _sorted_images(self, images: list[tuple[str, float]]) -> list[tuple[str, float]]:
        if self.sort_dropdown.value == "score":
            return sorted(images, key=lambda x: x[1], reverse=not self._sort_ascending)
        else:
            return sorted(images, key=lambda x: Path(x[0]).name, reverse=not self._sort_ascending)

    # --- 結果表示 ---

    def _load_results(self, output_dir: Path):
        if not output_dir.exists():
            return

        self._result_data.clear()
        for avatar_dir in sorted(output_dir.iterdir()):
            if not avatar_dir.is_dir():
                continue
            report_path = avatar_dir / "report.tsv"
            if not report_path.exists():
                continue
            images: list[tuple[str, float]] = []
            with open(report_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    images.append((row["path"], float(row["similarity"])))
            self._result_data[avatar_dir.name] = images

        self._rebuild_result_view()
        self._safe_update()

    def _build_grid(self, images: list[tuple[str, float]]) -> ft.GridView:
        grid = ft.GridView(
            runs_count=5,
            max_extent=180,
            child_aspect_ratio=0.95,
            spacing=2,
            run_spacing=2,
            expand=True,
        )
        for img_path, score in self._sorted_images(images):
            filename = Path(img_path).name
            grid.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Image(
                            src=img_path,
                            fit=ft.BoxFit.CONTAIN,
                            border_radius=ft.border_radius.all(2),
                            expand=True,
                        ),
                        ft.Text(
                            filename,
                            size=8,
                            text_align=ft.TextAlign.CENTER,
                            no_wrap=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                            color=ft.Colors.GREY_700,
                        ),
                        ft.Text(
                            f"{score:.4f}",
                            size=10,
                            text_align=ft.TextAlign.CENTER,
                            weight=ft.FontWeight.W_500,
                        ),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
                    on_click=lambda _e, p=img_path: self._open_image(p),
                    ink=True,
                    border_radius=ft.border_radius.all(4),
                    padding=2,
                )
            )
        return grid

    def _rebuild_result_view(self):
        if not self._result_data:
            self.result_container.content = ft.Text("結果が見つかりません", size=14, color=ft.Colors.GREY)
            return

        tab_labels: list[ft.Tab] = []
        tab_views: list[ft.Control] = []
        for avatar_name, images in self._result_data.items():
            tab_labels.append(ft.Tab(label=f"{avatar_name} ({len(images)})"))
            tab_views.append(self._build_grid(images))

        self.result_container.content = ft.Column([
            ft.Row([self.sort_dropdown, self.sort_order_button], spacing=4),
            ft.Tabs(
                content=ft.Column([
                    ft.TabBar(tabs=tab_labels),
                    ft.TabBarView(controls=tab_views, expand=True),
                ]),
                length=len(tab_labels),
                expand=True,
            ),
        ], spacing=4, expand=True)

    @staticmethod
    def _open_image(path: str):
        try:
            os.startfile(path)
        except Exception:
            pass


def main():
    # Windows の asyncio ProactorEventLoop 終了時の ConnectionResetError を抑制
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    ft.app(target=VsfGui)
    os._exit(0)
