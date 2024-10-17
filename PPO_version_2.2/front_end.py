import sys
import os
import importlib.util
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QTextEdit, QPushButton, QLineEdit, QFileDialog, QScrollArea, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QWaitCondition, QMutex
import re  # For sanitizing the file name


class FileExecutorThread(QThread):
    output_signal = pyqtSignal(str)
    pause_signal = pyqtSignal(bool)

    def __init__(self, file_path, steps):
        super().__init__()
        self.file_path = file_path
        self.steps = steps
        self._pause = False
        self._pause_cond = QWaitCondition()
        self._mutex = QMutex()

    def run(self):
        try:
            spec = importlib.util.spec_from_file_location("module.name", self.file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'run'):
                self.output_signal.emit(f"Running {self.file_path} with steps {self.steps}...")
                for i in range(self.steps if self.steps else 10):  # Example of steps iteration
                    self._mutex.lock()
                    while self._pause:
                        self._pause_cond.wait(self._mutex)
                    self._mutex.unlock()

                    # Example logic for each step (replace with real logic from the script)
                    result = module.run(i) if hasattr(module, 'run') else None
                    self.output_signal.emit(f"Step {i}: {result}")
            else:
                self.output_signal.emit(f"Executed {self.file_path}, but no 'run' function found.")
        except Exception as e:
            error_message = f"Error during execution: {traceback.format_exc()}"
            self.output_signal.emit(error_message)

    def toggle_pause(self):
        """Pause or resume execution."""
        self._mutex.lock()
        self._pause = not self._pause
        if not self._pause:
            self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.pause_signal.emit(self._pause)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Python File Executor")
        self.setFixedSize(900, 600)

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Initialize scroll area but don't show buttons until a file is selected
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedSize(300, 600)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        # Create initial "Choose Model" button
        self.choose_model_button = QPushButton("Choose Model")
        self.choose_model_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.choose_model_button.clicked.connect(self.choose_model)

        # Add the "Choose Model" button to the layout
        self.scroll_layout.addWidget(self.choose_model_button)

        main_layout.addWidget(self.scroll_area)

        # Right side layout with text area and input steps
        right_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setFixedSize(500, 300)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(self.text_edit)

        self.input_steps = QLineEdit(self)
        self.input_steps.setPlaceholderText("Enter max steps")
        right_layout.addWidget(self.input_steps)

        button_layout = QHBoxLayout()

        # Start button
        start_button = QPushButton("Start")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        start_button.clicked.connect(self.start_execution)
        button_layout.addWidget(start_button)

        # Pause/Resume button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #ec971f;
            }
        """)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause_execution)
        button_layout.addWidget(self.pause_button)

        # Stop button
        stop_button = QPushButton("Stop")
        stop_button.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
        """)
        stop_button.clicked.connect(self.stop_execution)
        button_layout.addWidget(stop_button)

        right_layout.addLayout(button_layout)
        main_layout.addLayout(right_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.executor_thread = None
        self.selected_python_file = None  # Track the currently selected Python file
        self.file_buttons = {}  # Dictionary to store file paths and associated buttons

    def sanitize_variable_name(self, file_name):
        """Sanitize the file name to create a valid Python variable name."""
        sanitized_name = re.sub(r'\W|^(?=\d)', '_', file_name)  # Replace non-alphanumeric characters with underscores
        return sanitized_name

    def choose_model(self):
    #Open file dialog to select any file.
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Any File", "", "All Files (*)", options=options)
        if file_name:
            self.add_file_button(file_name)


    def add_file_button(self, file_path):
        """Add a button for the selected file and store its path."""
        file_name = os.path.basename(file_path)
        variable_name = self.sanitize_variable_name(os.path.splitext(file_name)[0])  # Remove extension and sanitize

        # Check if the file is already loaded
        if variable_name in globals():
            self.text_edit.append(f"Model {file_name} is already loaded as variable {variable_name}.")
            return

        # Create a global variable with the sanitized name
        globals()[variable_name] = file_path
        self.text_edit.append(f"Created variable '{variable_name}' for file: {file_path}")

        # Create a layout for each variable with its name, input, and buttons
        var_layout = QVBoxLayout()

        # Label for showing variable name and current value
        label = QLabel(f"Variable: {variable_name}")
        var_layout.addWidget(label)

        # Input field to modify the variable (file path)
        var_input = QLineEdit(self)
        var_input.setText(file_path)
        var_layout.addWidget(var_input)

        # Update button to modify the variable's value
        update_button = QPushButton("Update Variable")
        update_button.clicked.connect(lambda: self.update_variable(variable_name, var_input.text()))
        var_layout.addWidget(update_button)

        # Run button to execute the file linked to the variable
        run_button = QPushButton(f"Run {variable_name}")
        run_button.clicked.connect(lambda: self.file_clicked(file_path))
        var_layout.addWidget(run_button)

        # Clear button to remove the variable
        clear_button = QPushButton(f"Clear {variable_name}")
        clear_button.clicked.connect(lambda: self.clear_variable(variable_name, var_layout))
        var_layout.addWidget(clear_button)

        # Add the variable layout to the scroll area
        self.scroll_layout.addLayout(var_layout)
        self.file_buttons[file_name] = file_path

    def update_variable(self, var_name, new_value):
        """Update the value of the variable and display in the text edit."""
        globals()[var_name] = new_value
        self.text_edit.append(f"Updated variable '{var_name}' to new value: {new_value}")

    def clear_variable(self, var_name, layout):
        """Clear the variable and remove its layout from the UI."""
        if var_name in globals():
            del globals()[var_name]
            self.text_edit.append(f"Cleared variable '{var_name}'")

        # Hide the layout from the UI
        for i in reversed(range(layout.count())): 
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def file_clicked(self, file_path):
        """Handle file button click and set selected Python file."""
        self.text_edit.append(f"Selected: {file_path}")
        self.selected_python_file = file_path

    def start_execution(self):
        """Start execution of the selected Python file."""
        if self.selected_python_file:
            steps = self.input_steps.text()
            if steps.isdigit():
                steps = int(steps)
            else:
                steps = None

            self.text_edit.append(f"Executing {self.selected_python_file} with steps: {steps}...")

            self.executor_thread = FileExecutorThread(self.selected_python_file, steps)
            self.executor_thread.output_signal.connect(self.append_output)
            self.executor_thread.start()

            # Enable pause button after starting
            self.pause_button.setEnabled(True)
        else:
            self.text_edit.append("No valid file selected.")

    def append_output(self, output):
        """Display output in the text edit."""
        self.text_edit.append(output)

    def toggle_pause_execution(self):
        """Toggle pause/resume in the thread."""
        if self.executor_thread:
            self.executor_thread.toggle_pause()

            # Update pause button text based on the state
            if self.executor_thread._pause:
                self.pause_button.setText("Resume")
            else:
                self.pause_button.setText("Pause")

    def stop_execution(self):
        """Stop the current execution, clear the output, and reset the selected file."""
        # Clear the output area
        self.text_edit.clear()
        self.text_edit.append("Output cleared.")

        # Reset the selected Python file
        self.selected_python_file = None
        self.text_edit.append("Selected file has been cleared.")

        # Reset pause button
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)  # Disable the pause button when stopped


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
