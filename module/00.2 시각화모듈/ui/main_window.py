from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QMessageBox
from utils.geometry import calculate_lines

class MainWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Unique ID Plot Viewer")
        self.setGeometry(100, 100, 400, 400)
        self.df = df

        layout = QVBoxLayout()

        label = QLabel("Select a unique_id to view the plot:")
        layout.addWidget(label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search unique_id")
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.unique_ids = sorted(df['UNIQUE_ID'].unique())
        self.list_widget.addItems(self.unique_ids)

        self.search_box.textChanged.connect(self.filter_unique_ids)

        button = QPushButton("Show Plot")
        button.clicked.connect(self.show_plot)
        layout.addWidget(button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def filter_unique_ids(self, text):
        self.list_widget.clear()
        filtered_ids = [uid for uid in self.unique_ids if text.lower() in uid.lower()]
        self.list_widget.addItems(filtered_ids)

    def show_plot(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            unique_id = selected_items[0].text()
            df_lot = self.df[self.df['UNIQUE_ID'] == unique_id]

            # 기존 코드에서 step_pitch 및 map_shift를 사용해 die라인 계산
            step_pitch_x = df_lot['STEP_PITCH_X'].iloc[0]
            step_pitch_y = df_lot['STEP_PITCH_Y'].iloc[0]
            map_shift_x = df_lot['MAP_SHIFT_X'].iloc[0]
            map_shift_y = df_lot['MAP_SHIFT_Y'].iloc[0]
            max_die_x = int(df_lot['DieX'].max())
            min_die_x = int(df_lot['DieX'].min())
            max_die_y = int(df_lot['DieY'].max())
            min_die_y = int(df_lot['DieY'].min())

            start_left = -(step_pitch_x)/2 + map_shift_x
            start_bottom = -(step_pitch_y)/2 + map_shift_y

            vertical_lines = calculate_lines(start_left, step_pitch_x, max_die_x, min_die_x)
            horizontal_lines = calculate_lines(start_bottom, step_pitch_y, max_die_y, min_die_y)

            from plotting.plot_windows import PlotWindow
            self.plot_window = PlotWindow(unique_id, self.df, vertical_lines, horizontal_lines)
            self.plot_window.show()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a unique_id from the list.")
