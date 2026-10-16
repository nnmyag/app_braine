# -*- coding: utf-8 -*-
"""
PyQt5 GUI for finding and deleting files/folders whose names contain given keyword(s).

Workflow:
    1. Select the master folder to scan.
    2. Enter keyword(s) (comma- or space-separated) and check the object types
       to look for (files, folders).
    3. Scan: lists every object whose name contains ALL keywords, so the user
       can double-check the result before anything is touched.
    4. Delete: asks for a single confirmation ("delete N objects?"), then
       performs the deletion.

Notes:
    - Matching is done on the object NAME only (not the full path).
    - Folders are deleted recursively (shutil.rmtree).
    - Nested matches are pruned: if a matched folder contains other matched
      objects, only the top-most folder is deleted (children go with it).

Dependencies:
    pip install pyqt5
"""

import os
import sys
import shutil

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QCheckBox,
    QFileDialog, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QProgressBar, QAbstractItemView
)


# =========================
# Helpers
# =========================
# Pre-filled in the keyword box; edit or clear it in the window as needed.
DEFAULT_KEYWORDS = "__pycache__"


def parse_keywords(text):
    """Split a raw keyword string on commas / whitespace into a clean list."""
    return [k for k in text.replace(",", " ").split() if k]


def name_matches(name, keywords, case_sensitive):
    """True if `name` contains every keyword."""
    target = name if case_sensitive else name.lower()
    keys = keywords if case_sensitive else [k.lower() for k in keywords]
    return all(k in target for k in keys)


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0 or unit == "TB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0


def folder_size(path):
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def prune_nested(paths):
    """Drop any path that lives inside another matched path (parents win)."""
    kept = []
    for p in sorted(paths, key=lambda x: x.count(os.sep)):
        norm = os.path.normpath(p)
        if any(norm.startswith(os.path.normpath(k) + os.sep) for k in kept):
            continue
        kept.append(norm)
    return kept


# =========================
# Scan worker
# =========================
class ScanWorker(QThread):
    """Walks the master folder in the background and reports every match."""

    found = pyqtSignal(str, str, int)      # path, kind ("File"/"Folder"), size
    progress = pyqtSignal(str)             # currently scanned directory
    finished_scan = pyqtSignal(int)        # number of matches

    def __init__(self, root, keywords, want_files, want_folders, case_sensitive):
        super().__init__()
        self.root = root
        self.keywords = keywords
        self.want_files = want_files
        self.want_folders = want_folders
        self.case_sensitive = case_sensitive
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        n = 0
        for dirpath, dirnames, filenames in os.walk(self.root, onerror=lambda e: None):
            if self._stop:
                break
            self.progress.emit(dirpath)

            if self.want_folders:
                for d in list(dirnames):
                    if self._stop:
                        break
                    if name_matches(d, self.keywords, self.case_sensitive):
                        full = os.path.join(dirpath, d)
                        self.found.emit(full, "Folder", folder_size(full))
                        n += 1

            if self.want_files:
                for f in filenames:
                    if self._stop:
                        break
                    if name_matches(f, self.keywords, self.case_sensitive):
                        full = os.path.join(dirpath, f)
                        try:
                            size = os.path.getsize(full)
                        except OSError:
                            size = 0
                        self.found.emit(full, "File", size)
                        n += 1

        self.finished_scan.emit(n)


# =========================
# Main GUI
# =========================
class FindAndDeleteGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.master_folder = ""
        self.worker = None
        self._init_ui()

    # ---- UI construction ------------------------------------------------
    def _init_ui(self):
        self.setWindowTitle("Find & Delete Objects by Keyword")
        self.resize(950, 620)

        layout = QVBoxLayout(self)

        # --- Step 1: master folder
        row_folder = QHBoxLayout()
        self.btn_browse = QPushButton("1. Select master folder...")
        self.btn_browse.clicked.connect(self.select_folder)
        self.lbl_folder = QLabel("<i>No folder selected</i>")
        self.lbl_folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_folder.addWidget(self.btn_browse)
        row_folder.addWidget(self.lbl_folder, 1)
        layout.addLayout(row_folder)

        # --- Step 2: keywords + object types
        row_kw = QHBoxLayout()
        row_kw.addWidget(QLabel("2. Keyword(s):"))
        self.edit_keywords = QLineEdit(DEFAULT_KEYWORDS)
        self.edit_keywords.setPlaceholderText(
            "separate by comma or space - an object must contain ALL of them")
        self.edit_keywords.returnPressed.connect(self.start_scan)
        row_kw.addWidget(self.edit_keywords, 1)
        layout.addLayout(row_kw)

        row_opt = QHBoxLayout()
        self.chk_files = QCheckBox("Files")
        self.chk_files.setChecked(True)
        self.chk_folders = QCheckBox("Folders")
        self.chk_folders.setChecked(True)
        self.chk_case = QCheckBox("Case sensitive")
        row_opt.addWidget(QLabel("Object types:"))
        row_opt.addWidget(self.chk_files)
        row_opt.addWidget(self.chk_folders)
        row_opt.addSpacing(20)
        row_opt.addWidget(self.chk_case)
        row_opt.addStretch(1)
        self.btn_scan = QPushButton("3. Scan")
        self.btn_scan.clicked.connect(self.start_scan)
        row_opt.addWidget(self.btn_scan)
        layout.addLayout(row_opt)

        # --- Result list
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Type", "Name", "Size", "Path"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setColumnWidth(0, 70)
        self.tree.setColumnWidth(1, 250)
        self.tree.setColumnWidth(2, 90)
        layout.addWidget(self.tree, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)         # busy indicator
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("Ready.")
        layout.addWidget(self.lbl_status)

        # --- Step 4: delete
        row_del = QHBoxLayout()
        self.btn_clear = QPushButton("Clear list")
        self.btn_clear.clicked.connect(self.clear_results)
        self.btn_delete = QPushButton("4. Delete listed objects")
        self.btn_delete.setStyleSheet("font-weight: bold; color: #a00000;")
        self.btn_delete.clicked.connect(self.delete_objects)
        self.btn_delete.setEnabled(False)
        row_del.addWidget(self.btn_clear)
        row_del.addStretch(1)
        row_del.addWidget(self.btn_delete)
        layout.addLayout(row_del)

    # ---- Step 1 ---------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select master folder", self.master_folder or os.path.expanduser("~"))
        if folder:
            self.master_folder = folder
            self.lbl_folder.setText(folder)
            self.clear_results()

    # ---- Step 3 ---------------------------------------------------------
    def start_scan(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            return

        if not self.master_folder or not os.path.isdir(self.master_folder):
            QMessageBox.warning(self, "No folder", "Please select a master folder first.")
            return

        keywords = parse_keywords(self.edit_keywords.text())
        if not keywords:
            QMessageBox.warning(self, "No keyword", "Please enter at least one keyword.")
            return

        if not (self.chk_files.isChecked() or self.chk_folders.isChecked()):
            QMessageBox.warning(self, "No object type",
                                "Please check at least one object type (Files / Folders).")
            return

        self.clear_results()
        self.progress.setVisible(True)
        self.btn_scan.setText("Stop")
        self.btn_delete.setEnabled(False)

        self.worker = ScanWorker(
            self.master_folder, keywords,
            self.chk_files.isChecked(), self.chk_folders.isChecked(),
            self.chk_case.isChecked())
        self.worker.found.connect(self.add_result)
        self.worker.progress.connect(
            lambda p: self.lbl_status.setText(f"Scanning: {p}"))
        self.worker.finished_scan.connect(self.scan_done)
        self.worker.start()

    def add_result(self, path, kind, size):
        item = QTreeWidgetItem([kind, os.path.basename(path), human_size(size), path])
        item.setData(0, Qt.UserRole, path)
        self.tree.addTopLevelItem(item)

    def scan_done(self, n):
        self.progress.setVisible(False)
        self.btn_scan.setText("3. Scan")
        self.btn_delete.setEnabled(n > 0)
        self.lbl_status.setText(
            f"Found {n} object(s). Please double-check the list before deleting.")

    def clear_results(self):
        self.tree.clear()
        self.btn_delete.setEnabled(False)
        self.lbl_status.setText("Ready.")

    # ---- Step 4 ---------------------------------------------------------
    def _listed_paths(self):
        return [self.tree.topLevelItem(i).data(0, Qt.UserRole)
                for i in range(self.tree.topLevelItemCount())]

    def delete_objects(self):
        paths = self._listed_paths()
        if not paths:
            return

        targets = prune_nested(paths)
        n_shown = len(paths)
        n_real = len(targets)

        msg = f"Delete {n_shown} object(s) listed below the master folder?\n\n{self.master_folder}"
        if n_real != n_shown:
            msg += (f"\n\n({n_shown - n_real} of them are nested inside another matched "
                    f"folder and will be removed together with their parent, "
                    f"so {n_real} deletion(s) will be performed.)")
        msg += "\n\nThis operation CANNOT be undone."

        reply = QMessageBox.question(
            self, "Confirm deletion", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            self.lbl_status.setText("Deletion cancelled.")
            return

        deleted, errors = 0, []
        for p in targets:
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                elif os.path.exists(p) or os.path.islink(p):
                    os.remove(p)
                else:
                    continue          # already gone with a parent
                deleted += 1
            except Exception as exc:
                errors.append(f"{p}\n    -> {exc}")

        self.clear_results()
        if errors:
            QMessageBox.warning(
                self, "Deletion finished with errors",
                f"Deleted {deleted} object(s).\n\nFailed on {len(errors)}:\n\n"
                + "\n".join(errors[:15])
                + ("\n..." if len(errors) > 15 else ""))
        else:
            QMessageBox.information(self, "Done", f"Deleted {deleted} object(s).")
        self.lbl_status.setText(f"Deleted {deleted} object(s).")

    # ---- housekeeping ---------------------------------------------------
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = FindAndDeleteGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
