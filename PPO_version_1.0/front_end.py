import sys
import os
from back_end import main
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QTextEdit, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QSize, Qt, pyqtSignal


class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)  # Define a signal that passes a string parameter

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.label_text = ""

    def mousePressEvent(self, event):
        self.clicked.emit(self.label_text)  # Emit the signal with the label text
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window title
        self.setWindowTitle("Function Description Window")

        # Set window size and fix it
        self.setFixedSize(800, 600)

        # Create main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create a scroll area and set fixed size
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedSize(300, 600)  # Limit content to 300x600 area
        scroll_area.setStyleSheet(""" 
            QScrollBar:vertical {
                border: none;
                background-color: #e0e0e0;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #909090;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #606060;
            }
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                height: 0px;
            }
        """)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 只保留两个按钮及其图像描述
        function_configs = [
            ("belt: Drag belt:", "belt.png"),
            ("Building: Place building:", "cutter.png"),
        ]

        # Get the path to the image folder
        image_folder = os.path.join(os.getcwd(), "button_image")

        # Create text edit area on the right side
        self.text_edit = QTextEdit()
        self.text_edit.setFixedSize(500, 300)  # Set text box size to 500x300
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show vertical scroll bar
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Always show horizontal scroll bar
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        # Input fields for x and y coordinates
        self.input_x = QLineEdit(self)
        self.input_x.setPlaceholderText("Enter X coordinate")
        self.input_y = QLineEdit(self)
        self.input_y.setPlaceholderText("Enter Y coordinate")

        # Store the selected image name to execute later
        self.selected_image_name = None

        # 用于记录点击的按钮，用于在点击 Start 时调用正确的函数
        self.selected_function = None

        # Add function descriptions and images (只创建两个按钮)
        for description, image_name in function_configs:
            h_layout = QHBoxLayout()

            # Create image container and set fixed size
            image_container = QWidget()
            image_container.setFixedSize(100, 100)
            image_layout = QVBoxLayout(image_container)
            image_layout.setContentsMargins(0, 0, 0, 0)
            image_layout.setAlignment(Qt.AlignCenter)

            # Add image
            image_path = os.path.join(image_folder, image_name)
            pixmap = QPixmap(image_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label = ClickableLabel()
            image_label.setPixmap(pixmap)
            image_label.label_text = description  # Assign description information
            image_label.setStyleSheet("border: 1px solid #ccc; border-radius: 10px; padding: 5px;")  # Enhance image display
            image_label.clicked.connect(self.image_clicked)  # Connect the click signal to the slot function
            image_layout.addWidget(image_label)

            h_layout.addWidget(image_container)

            # Add function description
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)  # Allow text wrapping
            desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)  # Allow description label to expand
            desc_label.setAlignment(Qt.AlignVCenter)  # Center text vertically
            desc_label.setStyleSheet("padding-left: 10px; padding-top: 5px; padding-bottom: 5px;")  # Add padding
            h_layout.addWidget(desc_label)

            # Add to the vertical layout
            scroll_layout.addLayout(h_layout)

        # Set scroll area content
        scroll_area.setWidget(scroll_content)

        # Add scroll area to the left side of the main layout
        main_layout.addWidget(scroll_area)

        # Add a vertical layout to the right side
        right_layout = QVBoxLayout()

        # Add the text box and input fields to the top of the right layout
        right_layout.addWidget(self.text_edit)
        right_layout.addWidget(self.input_x)
        right_layout.addWidget(self.input_y)

        # Add an information display label and two buttons to the bottom-right corner
        info_layout = QVBoxLayout()
        info_display = QLabel("Information Display")
        info_display.setFixedSize(500, 200)  # Set information display size
        info_display.setAlignment(Qt.AlignCenter)
        info_display.setStyleSheet("""
            background-color: #f5f5f5;
            border: 1px solid #bbb;
            border-radius: 10px;
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #ffffff, stop:1 #e0e0e0);
        """)  # Use a smoother gradient background for the information display area
        info_layout.addWidget(info_display)

        # Add buttons
        button_layout = QHBoxLayout()
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
        """)  # Hover effect
        start_button.clicked.connect(self.start_execution)  # Connect the start button to the execution function
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
        """)  # Hover effect
        button_layout.addWidget(start_button)
        button_layout.addWidget(stop_button)
        info_layout.addLayout(button_layout)

        # Add the information display and button layout to the bottom of the right layout
        right_layout.addLayout(info_layout)

        # Add the right layout to the main layout
        main_layout.addLayout(right_layout)

        # Set layout to the main widget
        central_widget.setLayout(main_layout)

        # Set the main widget as the central widget of the window
        self.setCentralWidget(central_widget)


    def image_clicked(self, description):
        self.text_edit.append(description)  # Display the clicked button description in the text box
        self.selected_image_name = description.split(":")[0].lower()  # Store the clicked image name
        
        # 根据按钮的描述决定要调用的函数
        if "belt" in self.selected_image_name:
            self.selected_function = "run_drag_belt_process"
        elif "building" in self.selected_image_name:
            self.selected_function = "run_place_single_object"
        else:
            self.selected_function = None

    def start_execution(self):
        # 根据点击的按钮执行不同的函数
        if self.selected_function == "run_drag_belt_process":
            main()
        elif self.selected_function == "run_place_single_object":
            main()  # 调用 run_place_single_object
        else:
            self.text_edit.append("No valid function selected.")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
