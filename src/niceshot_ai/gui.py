import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.process = None
        self.analysis_running = False
        self.close_requested = False

        # ---------------------------------------------------------
        # Window
        # ---------------------------------------------------------

        self.setWindowTitle("NiceShot AI")
        self.setMinimumSize(600, 600)
        self.resize(600, 600)

        # ---------------------------------------------------------
        # Application paths
        # ---------------------------------------------------------

        if getattr(sys, "frozen", False):
            self.base_dir = Path(sys.executable).resolve().parent
        else:
            self.base_dir = Path(__file__).resolve().parent

        self.root_dir = self.base_dir.parent.parent.parent

        icon_path = self.base_dir / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # ---------------------------------------------------------
        # Colors
        # ---------------------------------------------------------

        self.colors = {
            "background": "#0D1117",
            "surface": "#161B22",
            "surface2": "#1C2128",
            "border": "#30363D",
            "text": "#F0F6FC",
            "muted": "#8B949E",
            "primary": "#7C3AED",
            "primary_hover": "#8B5CF6",
            "accent": "#00D4FF",
            "success": "#3FB950",
            "danger": "#F85149",
            "input": "#0D1117",
        }

        self.setup_style()
        self.setup_ui()

    # =============================================================
    # Styling
    # =============================================================

    def setup_style(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {self.colors["background"]};
            }}

            QWidget {{
                color: {self.colors["text"]};
                font-family: "Segoe UI";
                font-size: 9pt;
            }}

            QLabel {{
                background: transparent;
            }}

            QLabel#Title {{
                font-size: 19pt;
                font-weight: 700;
                color: {self.colors["text"]};
            }}

            QLabel#Subtitle {{
                font-size: 8pt;
                color: {self.colors["muted"]};
            }}

            QLabel#SectionTitle {{
                font-size: 8pt;
                font-weight: 700;
                color: {self.colors["muted"]};
            }}

            QLabel#StatusLabel {{
                color: {self.colors["success"]};
                font-size: 8pt;
                font-weight: 600;
            }}

            QFrame#Card {{
                background-color: {self.colors["surface"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 9px;
            }}

            QFrame#HeaderCard {{
                background-color: {self.colors["surface"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 9px;
            }}

            QLineEdit {{
                background-color: {self.colors["input"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 6px 9px;
                color: {self.colors["text"]};
                selection-background-color: {self.colors["primary"]};
                min-height: 18px;
            }}

            QLineEdit:focus {{
                border: 1px solid {self.colors["primary"]};
            }}

            QComboBox {{
                background-color: {self.colors["input"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 6px 9px;
                color: {self.colors["text"]};
                min-height: 18px;
            }}

            QComboBox:hover {{
                border: 1px solid {self.colors["primary"]};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 25px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {self.colors["surface"]};
                color: {self.colors["text"]};
                border: 1px solid {self.colors["border"]};
                selection-background-color: {self.colors["primary"]};
                selection-color: white;
            }}

            QPushButton {{
                background-color: {self.colors["surface2"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 6px 11px;
                color: {self.colors["text"]};
                font-weight: 600;
            }}

            QPushButton:hover {{
                background-color: {self.colors["border"]};
            }}

            QPushButton:pressed {{
                background-color: #252C35;
            }}

            QPushButton:disabled {{
                color: {self.colors["muted"]};
                background-color: #15191F;
            }}

            QPushButton#BrowseButton {{
                min-width: 65px;
                min-height: 18px;
            }}

            QPushButton#AnalyzeButton {{
                background-color: {self.colors["primary"]};
                border: none;
                border-radius: 7px;
                padding: 9px 15px;
                font-size: 9pt;
                font-weight: 700;
                color: white;
            }}

            QPushButton#AnalyzeButton:hover {{
                background-color: {self.colors["primary_hover"]};
            }}

            QPushButton#AnalyzeButton:disabled {{
                background-color: #39205F;
                color: #A78BCA;
            }}

            QCheckBox {{
                spacing: 6px;
                color: {self.colors["text"]};
                padding: 1px;
                font-size: 8.5pt;
            }}

            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
                border-radius: 4px;
                border: 1px solid {self.colors["border"]};
                background-color: {self.colors["input"]};
            }}

            QCheckBox::indicator:hover {{
                border: 1px solid {self.colors["primary"]};
            }}

            QCheckBox::indicator:checked {{
                background-color: {self.colors["primary"]};
                border: 1px solid {self.colors["primary"]};
            }}

            QSpinBox {{
                background-color: {self.colors["input"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 6px;
                padding: 5px 7px;
                color: {self.colors["text"]};
                min-height: 18px;
            }}

            QSpinBox:focus {{
                border: 1px solid {self.colors["primary"]};
            }}

            QProgressBar {{
                background-color: {self.colors["input"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 5px;
                height: 9px;
                text-align: center;
                color: white;
            }}

            QProgressBar::chunk {{
                background-color: {self.colors["primary"]};
                border-radius: 4px;
            }}
            """
        )

    # =============================================================
    # UI
    # =============================================================

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Compact outer layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(9)

        # =========================================================
        # HEADER
        # =========================================================

        header = QFrame()
        header.setObjectName("HeaderCard")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(13, 8, 13, 8)
        header_layout.setSpacing(5)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)

        title = QLabel("NiceShot AI")
        title.setObjectName("Title")

        subtitle = QLabel("AI-powered gameplay analysis")
        subtitle.setObjectName("Subtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        self.status_label = QLabel("● Ready")
        self.status_label.setObjectName("StatusLabel")

        header_layout.addWidget(self.status_label)

        main_layout.addWidget(header)

        # =========================================================
        # FILES CARD
        # =========================================================

        files_card = QFrame()
        files_card.setObjectName("Card")

        files_layout = QVBoxLayout(files_card)
        files_layout.setContentsMargins(13, 10, 13, 12)
        files_layout.setSpacing(6)

        # Game
        game_label = QLabel("GAME")
        game_label.setObjectName("SectionTitle")
        files_layout.addWidget(game_label)

        self.combo1 = QComboBox()
        self.combo1.addItems(
            [
                "Call of Duty: Black Ops 6",
                "Call of Duty: Black Ops 7",
            ]
        )
        self.combo1.setCurrentIndex(0)

        files_layout.addWidget(self.combo1)

        # Input
        input_label = QLabel("GAMEPLAY VIDEO")
        input_label.setObjectName("SectionTitle")
        files_layout.addWidget(input_label)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)

        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Select gameplay video...")
        self.input_entry.setMinimumHeight(32)

        input_button = QPushButton("Browse")
        input_button.setObjectName("BrowseButton")
        input_button.clicked.connect(self.browse_input)

        input_layout.addWidget(self.input_entry, 1)
        input_layout.addWidget(input_button, 0)

        files_layout.addLayout(input_layout)

        # Output
        output_label = QLabel("OUTPUT FOLDER")
        output_label.setObjectName("SectionTitle")
        files_layout.addWidget(output_label)

        output_layout = QHBoxLayout()
        output_layout.setSpacing(5)

        self.output_entry = QLineEdit()
        self.output_entry.setPlaceholderText("Select output folder...")
        self.output_entry.setMinimumHeight(32)

        output_button = QPushButton("Browse")
        output_button.setObjectName("BrowseButton")
        output_button.clicked.connect(self.browse_output)

        output_layout.addWidget(self.output_entry, 1)
        output_layout.addWidget(output_button, 0)

        files_layout.addLayout(output_layout)

        main_layout.addWidget(files_card)

        # =========================================================
        # OPTIONS CARD
        # =========================================================

        options_card = QFrame()
        options_card.setObjectName("Card")

        options_layout = QVBoxLayout(options_card)
        options_layout.setContentsMargins(13, 10, 13, 11)
        options_layout.setSpacing(7)

        options_title = QLabel("ANALYSIS OPTIONS")
        options_title.setObjectName("SectionTitle")
        options_layout.addWidget(options_title)

        # Checkboxes
        checkbox_grid = QGridLayout()
        checkbox_grid.setHorizontalSpacing(12)
        checkbox_grid.setVerticalSpacing(3)

        self.save_clips = QCheckBox("Save Clips")
        self.create_compilation = QCheckBox("Create Compilation")
        self.vertical_format = QCheckBox("Vertical format")
        self.analysis = QCheckBox("Session analysis")

        checkbox_grid.addWidget(self.save_clips, 0, 0)
        checkbox_grid.addWidget(self.create_compilation, 0, 1)
        checkbox_grid.addWidget(self.vertical_format, 1, 0)
        checkbox_grid.addWidget(self.analysis, 1, 1)

        options_layout.addLayout(checkbox_grid)

        # Coaching
        coaching_layout = QHBoxLayout()
        coaching_layout.setSpacing(8)

        coaching_label = QLabel("Coaching")
        coaching_label.setMinimumWidth(65)

        self.combo2 = QComboBox()
        self.combo2.addItems(
            [
                "None",
                "Quick",
                "Basic",
                "Long",
                "Full",
            ]
        )
        self.combo2.setCurrentIndex(0)
        self.combo2.setMaximumWidth(180)

        coaching_layout.addWidget(coaching_label)
        coaching_layout.addWidget(self.combo2)
        coaching_layout.addStretch()

        options_layout.addLayout(coaching_layout)

        # Compilation length
        comp_layout = QHBoxLayout()
        comp_layout.setSpacing(8)

        comp_label = QLabel("Compilation")
        comp_label.setMinimumWidth(65)

        self.spin = QSpinBox()
        self.spin.setRange(0, 100)
        self.spin.setValue(0)
        self.spin.setSuffix(" min")
        self.spin.setMaximumWidth(100)

        comp_layout.addWidget(comp_label)
        comp_layout.addWidget(self.spin)
        comp_layout.addStretch()

        options_layout.addLayout(comp_layout)

        main_layout.addWidget(options_card)

        # =========================================================
        # START BUTTON
        # =========================================================

        self.analyze_btn = QPushButton(
            "▶   START GAMEPLAY ANALYSIS"
        )
        self.analyze_btn.setObjectName("AnalyzeButton")
        self.analyze_btn.setMinimumHeight(38)
        self.analyze_btn.clicked.connect(self.analyze_video)

        main_layout.addWidget(self.analyze_btn)

        # =========================================================
        # PROGRESS CARD
        # =========================================================

        progress_card = QFrame()
        progress_card.setObjectName("Card")

        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(13, 9, 13, 10)
        progress_layout.setSpacing(5)

        progress_header = QHBoxLayout()

        progress_title = QLabel("ANALYSIS PROGRESS")
        progress_title.setObjectName("SectionTitle")

        self.progress_percent = QLabel("0%")
        self.progress_percent.setStyleSheet(
            f"color: {self.colors['accent']}; font-weight: 700;"
        )

        progress_header.addWidget(progress_title)
        progress_header.addStretch()
        progress_header.addWidget(self.progress_percent)

        progress_layout.addLayout(progress_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        progress_layout.addWidget(self.progress_bar)

        self.progress_text = QLabel("Waiting...")
        self.progress_text.setStyleSheet(
            f"color: {self.colors['muted']}; font-size: 8pt;"
        )

        progress_layout.addWidget(self.progress_text)

        main_layout.addWidget(progress_card)

        # Don't add a large stretch here.
        # The layout should stay compact inside a 600x600 window.

    # =============================================================
    # Browse
    # =============================================================

    def browse_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select gameplay video",
            "",
            "Video files (*.mp4 *.mov *.avi);;All files (*.*)",
        )

        if filename:
            self.input_entry.setText(filename)

    def browse_output(self):
        foldername = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
        )

        if foldername:
            self.output_entry.setText(foldername)

    # =============================================================
    # Validation
    # =============================================================

    def validate_inputs(self):
        input_path = self.input_entry.text().strip()
        output_path = self.output_entry.text().strip()

        if not input_path or not os.path.isfile(input_path):
            QMessageBox.critical(
                self,
                "Invalid Input",
                "Please select a valid input video.",
            )
            return False

        if not output_path or not os.path.isdir(output_path):
            QMessageBox.critical(
                self,
                "Invalid Output",
                "Please select a valid output folder.",
            )
            return False

        coaching_type = self.combo2.currentText().lower()

        if (
            not self.save_clips.isChecked()
            and not self.create_compilation.isChecked()
            and not self.analysis.isChecked()
            and coaching_type == "none"
        ):
            QMessageBox.warning(
                self,
                "No Analysis Selected",
                "Please select at least one analysis option.",
            )
            return False

        return True

    # =============================================================
    # Start Analysis
    # =============================================================

    def analyze_video(self):
        if self.analysis_running:
            return

        if not self.validate_inputs():
            return

        input_path = self.input_entry.text().strip()
        output_path = self.output_entry.text().strip()

        save_clips = self.save_clips.isChecked()
        create_compilation = self.create_compilation.isChecked()
        vertical_format = self.vertical_format.isChecked()
        analysis = self.analysis.isChecked()

        chosen_game = self.combo1.currentText()

        montage_len_seconds = self.spin.value() * 60

        coaching_type = self.combo2.currentText().lower()

        # ---------------------------------------------------------
        # Reset progress
        # ---------------------------------------------------------

        self.progress_bar.setValue(0)
        self.progress_percent.setText("0%")
        self.progress_text.setText("Starting analysis...")

        self.status_label.setText("● Starting...")
        self.status_label.setStyleSheet(
            f"color: {self.colors['accent']}; font-weight: 600;"
        )

        self.analyze_btn.setDisabled(True)

        # ---------------------------------------------------------
        # Locate Python and CLI
        # ---------------------------------------------------------

        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path(__file__).resolve().parent

        root_dir = base_dir.parent.parent

        python_file = root_dir / "niceshot_env" / "Scripts" / "python.exe"
        cli_file = base_dir / "niceshot_ai_cli.py"

        # ---------------------------------------------------------
        # Check files
        # ---------------------------------------------------------

        if not python_file.exists():
            QMessageBox.critical(
                self,
                "Python Not Found",
                f"Could not find the Python executable:\n\n"
                f"{python_file}",
            )
            self.reset_after_error()
            return

        if not cli_file.exists():
            QMessageBox.critical(
                self,
                "CLI Not Found",
                f"Could not find niceshot_ai_cli.py:\n\n"
                f"{cli_file}",
            )
            self.reset_after_error()
            return

        # ---------------------------------------------------------
        # Build CLI arguments
        # ---------------------------------------------------------

        args = [
            str(cli_file),
            "--game",
            chosen_game,
            "--input",
            input_path,
            "--output",
            output_path,
            "--comp_len",
            str(montage_len_seconds),
        ]

        if vertical_format:
            args.append("--vertical_format")

        if analysis:
            args.append("--session_analysis")

        if save_clips:
            args.append("--save_clips")

        if create_compilation:
            args.append("--compilation")

        if coaching_type != "none":
            args.extend(
                [
                    "--coaching",
                    coaching_type,
                ]
            )

        # ---------------------------------------------------------
        # Start QProcess
        # ---------------------------------------------------------

        self.process = QProcess(self)

        self.process.setProgram(str(python_file))
        self.process.setArguments(args)

        self.process.started.connect(self.process_started)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)

        self.process.readyReadStandardOutput.connect(
            self.read_stdout
        )

        self.process.readyReadStandardError.connect(
            self.read_stderr
        )

        self.analysis_running = True

        self.process.start()

        # ---------------------------------------------------------
        # Progress timer
        # ---------------------------------------------------------

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)

    # =============================================================
    # Process callbacks
    # =============================================================

    def process_started(self):
        self.status_label.setText("● Analyzing")
        self.status_label.setStyleSheet(
            f"color: {self.colors['accent']}; font-weight: 600;"
        )

        self.progress_text.setText(
            "AI analysis is running..."
        )

    def read_stdout(self):
        if not self.process:
            return

        data = self.process.readAllStandardOutput()
        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        if text.strip():
            print(text, end="")

    def read_stderr(self):
        if not self.process:
            return

        data = self.process.readAllStandardError()
        text = bytes(data).decode(
            "utf-8",
            errors="replace",
        )

        if text.strip():
            print(text, end="")

    def process_error(self, error):
        if not self.analysis_running:
            return

        error_message = self.process.errorString()

        print("Process error:", error_message)

        self.status_label.setText("● Error")
        self.status_label.setStyleSheet(
            f"color: {self.colors['danger']}; font-weight: 600;"
        )

        QMessageBox.critical(
            self,
            "Analysis Error",
            "The analysis process could not be started.\n\n"
            f"{error_message}",
        )

        self.reset_after_error()

    # =============================================================
    # Progress
    # =============================================================

    def update_progress(self):
        if not self.process:
            return

        output_path = self.output_entry.text().strip()

        if not output_path:
            return

        progress_file = os.path.join(
            output_path,
            "progress.json",
        )

        if os.path.exists(progress_file):
            try:
                with open(
                    progress_file,
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                progress = data.get("PROGRESS", 0)
                msg = data.get("MSG", "")

                try:
                    progress = int(float(progress))
                except (TypeError, ValueError):
                    progress = 0

                progress = max(
                    0,
                    min(100, progress),
                )

                self.progress_bar.setValue(progress)
                self.progress_percent.setText(
                    f"{progress}%"
                )

                if msg:
                    self.progress_text.setText(msg)

            except (
                json.JSONDecodeError,
                OSError,
            ) as e:
                print(
                    "Error reading progress:",
                    e,
                )

    # =============================================================
    # Process Finished
    # =============================================================

    def process_finished(
        self,
        exit_code,
        exit_status,
    ):
        if hasattr(self, "progress_timer"):
            self.progress_timer.stop()

        self.analysis_running = False

        if self.close_requested:
            return

        if exit_code == 0:
            self.progress_bar.setValue(100)
            self.progress_percent.setText("100%")
            self.progress_text.setText(
                "Analysis complete!"
            )

            self.status_label.setText("● Complete")
            self.status_label.setStyleSheet(
                f"color: {self.colors['success']}; "
                "font-weight: 600;"
            )

            self.analyze_btn.setEnabled(True)

            output_path = self.output_entry.text().strip()

            QMessageBox.information(
                self,
                "Analysis Complete",
                "Analysis complete!\n\n"
                f"Results saved to:\n{output_path}",
            )

            self.cleanup_metadata(
                output_path
            )

        else:
            self.status_label.setText("● Failed")
            self.status_label.setStyleSheet(
                f"color: {self.colors['danger']}; "
                "font-weight: 600;"
            )

            self.progress_text.setText(
                f"Analysis failed "
                f"(exit code {exit_code})."
            )

            self.analyze_btn.setEnabled(True)

            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"The analysis process exited "
                f"with code {exit_code}.\n\n"
                "Check the CLI output/logs for "
                "more information.",
            )

    # =============================================================
    # Cleanup
    # =============================================================

    def cleanup_metadata(self, output_path):
        meta_files = (
            "status.json",
            "progress.json",
            "events_temp.json",
            "events_temp_2.json",
            "video1.csv",
            "timestamp_sorted.csv",
            "events_temp_3.json",
        )

        for filename in meta_files:
            file_path = os.path.join(
                output_path,
                filename,
            )

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError as e:
                print(
                    f"Could not remove "
                    f"{file_path}: {e}"
                )

    # =============================================================
    # Reset
    # =============================================================

    def reset_after_error(self):
        self.analysis_running = False

        if hasattr(self, "progress_timer"):
            self.progress_timer.stop()

        self.analyze_btn.setEnabled(True)

        self.status_label.setText("● Ready")
        self.status_label.setStyleSheet(
            f"color: {self.colors['success']}; "
            "font-weight: 600;"
        )

    # =============================================================
    # Close
    # =============================================================

    def closeEvent(self, event):
        if (
            self.process is not None
            and self.process.state()
            != QProcess.NotRunning
        ):
            reply = QMessageBox.question(
                self,
                "Analysis Running",
                "Gameplay analysis is still running.\n\n"
                "Do you want to stop it and close "
                "NiceShot AI?",
                QMessageBox.Yes
                | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                event.ignore()
                return

            self.close_requested = True

            if hasattr(
                self,
                "progress_timer",
            ):
                self.progress_timer.stop()

            self.process.terminate()

            if not self.process.waitForFinished(
                3000
            ):
                self.process.kill()
                self.process.waitForFinished(
                    2000
                )

        event.accept()


# =============================================================
# Main
# =============================================================

def main():
    app = QApplication(sys.argv)

    app.setApplicationName("NiceShot AI")
    app.setOrganizationName("NiceShot AI")

    window = GUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()