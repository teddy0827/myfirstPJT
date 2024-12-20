import sys
from ui.app import get_app_instance
from ui.main_window import MainWindow
from utils.data_loader import load_data
from PyQt5.QtWidgets import QFileDialog

def main():
    app = get_app_instance()
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if not file_path:
        sys.exit(0)

    try:
        df = load_data(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    main_window = MainWindow(df)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
