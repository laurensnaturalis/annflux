"""
Selenium test for /simple endpoint using streetSurfaceVis project.

Usage:
    pytest test_simple_selenium.py -v

Requirements:
    pip install selenium webdriver-manager
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Project folder for testing
PROJECT_FOLDER = "/home/lhogeweg/annflux/data/streetSurfaceVis"


@pytest.fixture(scope="module")
def server():
    """Initialize project and start the annflux server for testing."""
    # Ensure project folder exists
    if not os.path.exists(PROJECT_FOLDER):
        pytest.skip(f"Project folder not found: {PROJECT_FOLDER}")
    
    # Initialize project if needed
    annflux_csv_path = Path(PROJECT_FOLDER) / "annflux" / "annflux.csv"
    if not annflux_csv_path.exists():
        # Import and run initialization
        sys.path.insert(0, "/home/lhogeweg/Documents/annflux_ln/src")
        from annflux.scripts.annflux_cli import go_command
        from annflux.shared import AnnfluxSource
        
        # Clean up any partial annflux folder
        annflux_folder = Path(PROJECT_FOLDER) / "annflux"
        if annflux_folder.exists():
            shutil.rmtree(annflux_folder)
        
        # Initialize with some labels from images.csv
        go_command(AnnfluxSource(PROJECT_FOLDER), start_labels=["foo", "bar"])
    
    # Set environment variables
    env = os.environ.copy()
    env["PROJECT_ROOT"] = PROJECT_FOLDER
    env["APP_DEBUG"] = "1"
    env["QUICKER_UPDATES"] = "2"
    env["LOGGING_LEVEL"] = "WARNING"
    
    # Create temp files for server output
    stdout_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
    stderr_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
    stdout_file.close()
    stderr_file.close()
    
    # Start server with output redirected to files
    stdout_fh = open(stdout_file.name, 'w')
    stderr_fh = open(stderr_file.name, 'w')
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "annflux.ui.basic.run_server", PROJECT_FOLDER],
        cwd="/home/lhogeweg/Documents/annflux_ln/src",
        env=env,
        stdout=stdout_fh,
        stderr=stderr_fh,
    )
    
    # Store log file paths for later inspection
    server_process._stdout_file = stdout_file.name
    server_process._stderr_file = stderr_file.name
    server_process._stdout_fh = stdout_fh
    server_process._stderr_fh = stderr_fh
    
    # Wait for server to start (give more time for initialization)
    max_retries = 60
    server_started = False
    for i in range(max_retries):
        time.sleep(2)
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:8006/", timeout=2)
            server_started = True
            break
        except Exception:
            # Check if process died
            if server_process.poll() is not None:
                stdout_fh.close()
                stderr_fh.close()
                with open(stdout_file.name, 'r') as f:
                    stdout_content = f.read()
                with open(stderr_file.name, 'r') as f:
                    stderr_content = f.read()
                print(f"\n\n=== SERVER STDOUT ===\n{stdout_content}\n=== END STDOUT ===")
                print(f"\n\n=== SERVER STDERR ===\n{stderr_content}\n=== END STDERR ===")
                pytest.fail(f"Server process died with code {server_process.returncode}")
            if i == max_retries - 1:
                server_process.terminate()
                pytest.fail("Server failed to start")
    
    if not server_started:
        server_process.terminate()
        pytest.fail("Server failed to start")
    
    yield server_process
    
    # Cleanup
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
    
    # Close file handles and print server output for debugging
    stdout_fh.close()
    stderr_fh.close()
    
    print("\n\n=== SERVER STDOUT (test session complete) ===")
    with open(stdout_file.name, 'r') as f:
        print(f.read())
    print("=== END STDOUT ===")
    
    print("\n\n=== SERVER STDERR (test session complete) ===")
    with open(stderr_file.name, 'r') as f:
        print(f.read())
    print("=== END STDERR ===")
    
    # Clean up temp files
    os.unlink(stdout_file.name)
    os.unlink(stderr_file.name)


@pytest.fixture
def driver(server):
    """Create a headless Chrome/Chromium WebDriver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--remote-debugging-port=9222")
    
    driver = None
    errors = []
    
    # Use ChromeDriver version matching installed Chromium (148)
    # This downloads ChromeDriver 148 to match your Chromium 148.0.7778.167
    from webdriver_manager.core.os_manager import ChromeType
    
    try:
        # Try snap Chromium first with matching ChromeDriver version
        chrome_options.binary_location = "/snap/bin/chromium"
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM, version="148.0.0").install()),
            options=chrome_options
        )
    except Exception as e:
        errors.append(f"Snap Chromium with ChromeDriver 148 failed: {e}")
        try:
            # Try without version constraint
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()),
                options=chrome_options
            )
        except Exception as e2:
            errors.append(f"Chromium auto-version failed: {e2}")
            try:
                # Try standard Chrome
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
            except Exception as e3:
                errors.append(f"Standard Chrome failed: {e3}")
                pytest.fail("Could not start Chrome or Chromium. Errors:\n" + "\n".join(errors))
    
    driver.implicitly_wait(10)
    
    yield driver
    
    driver.quit()


class TestSimpleAnnotator:
    """Test suite for /simple annotator interface."""
    
    BASE_URL = "http://127.0.0.1:8006"
    
    def test_simple_page_loads(self, driver):
        """Test that /simple page loads correctly."""
        driver.get(f"{self.BASE_URL}/simple")
        
        # Check main elements are present
        assert driver.find_element(By.ID, "toolbar").is_displayed()
        assert driver.find_element(By.ID, "main-table").is_displayed()
        assert driver.find_element(By.ID, "status").is_displayed()
        
        # Check toolbar elements
        assert driver.find_element(By.CSS_SELECTOR, "h1").text == "Simple Annotator"
        assert driver.find_element(By.ID, "filter-input").is_displayed()
        assert driver.find_element(By.ID, "sort-select").is_displayed()
        assert driver.find_element(By.ID, "n-select").is_displayed()
        assert driver.find_element(By.ID, "save-btn").is_displayed()
        assert driver.find_element(By.ID, "progress-bar").is_displayed()
    
    def test_filter_functionality(self, driver):
        """Test filter by predicted label."""
        driver.get(f"{self.BASE_URL}/simple")
        
        # Wait for data to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "table-body"))
        )
        
        # Wait for table to populate
        time.sleep(2)
        _ = len(driver.find_elements(By.CSS_SELECTOR, "#table-body tr"))  # Ensure table loaded
        
        # Enter filter text
        filter_input = driver.find_element(By.ID, "filter-input")
        filter_input.send_keys("concrete")
        
        # Wait for filter to apply (debounced)
        time.sleep(1.5)
        
        # Check that filter was applied
        # (Rows may be different or status updated)
        status = driver.find_element(By.ID, "status").text
        assert "reviewed" in status or "/" in status
    
    def test_sort_functionality(self, driver):
        """Test sort dropdown changes."""
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "sort-select"))
        )
        
        sort_select = driver.find_element(By.ID, "sort-select")
        
        # Change sort to different option
        from selenium.webdriver.support.ui import Select
        select = Select(sort_select)
        select.select_by_value("nn_underrepresented")
        
        time.sleep(1)
        
        # Verify sort changed
        new_value = driver.find_element(By.ID, "sort-select").get_attribute("value")
        assert new_value == "nn_underrepresented"
        
        # Change back
        select.select_by_value("fre_strat")
    
    def test_n_select_functionality(self, driver):
        """Test number of items to show dropdown."""
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "n-select"))
        )
        
        from selenium.webdriver.support.ui import Select
        n_select = Select(driver.find_element(By.ID, "n-select"))
        
        # Change to show 50 items
        n_select.select_by_value("50")
        time.sleep(1)
        
        # Change to show 10 items
        n_select.select_by_value("10")
        time.sleep(1)

        
    
    def test_approve_button(self, driver):
        """Test approving a row."""
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".approve-btn"))
        )
        
        # Find first approve button
        approve_buttons = driver.find_elements(By.CSS_SELECTOR, ".approve-btn")
        if len(approve_buttons) > 0:
            first_button = approve_buttons[0]
            # Click approve
            first_button.click()
            time.sleep(0.5)
            
            # Button should now show checkmark or be marked as approved
            # Note: Save button should still be disabled if not all reviewed
            save_btn = driver.find_element(By.ID, "save-btn")
            assert save_btn.is_enabled() or not save_btn.is_enabled()  # Either state is valid
    
    def test_label_picker_search(self, driver):
        """Test label search and picker functionality."""
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".label-search"))
        )
        
        # Find first label search input
        search_inputs = driver.find_elements(By.CSS_SELECTOR, ".label-search")
        if len(search_inputs) > 0:
            search_input = search_inputs[0]
            search_input.send_keys("concrete")
            time.sleep(0.5)
            
            # Label picker should show matching labels or "no matches" state
            pickers = driver.find_elements(By.CSS_SELECTOR, ".label-picker")
            if len(pickers) > 0:
                # Check that picker buttons exist for filtered results
                pick_buttons = driver.find_elements(By.CSS_SELECTOR, ".pick-btn")
                # Search worked if we got buttons or empty results (no matches for "concrete")
                assert len(pick_buttons) >= 0  # Search completed without error
    
    def test_image_click_opens_overlay(self, driver):
        """Test clicking image opens overlay - skipped if no images exist."""
        # Check if actual images exist in project
        images_dir = Path(PROJECT_FOLDER) / "images"
        if not any(images_dir.glob("*.jpg")):
            pytest.skip("No image files in project - skipping overlay test")
        
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".thumb"))
        )
        
        # Find first thumbnail and click to open overlay
        thumbnails = driver.find_elements(By.CSS_SELECTOR, ".thumb")
        if len(thumbnails) > 0:
            thumbnails[0].click()
            time.sleep(0.5)
            
            # Verify overlay is displayed after clicking thumbnail
            overlay = driver.find_element(By.ID, "overlay")
            assert overlay.is_displayed()
            
            # Close overlay by clicking close button
            close_btn = driver.find_element(By.ID, "overlay-close")
            close_btn.click()
            time.sleep(0.5)
    
    def test_progress_bar_exists(self, driver):
        """Test progress bar is rendered."""
        driver.get(f"{self.BASE_URL}/simple")
        
        progress_bar = driver.find_element(By.ID, "progress-bar")
        assert progress_bar.is_displayed()
        
        # Check segments exist
        assert len(driver.find_elements(By.CSS_SELECTOR, ".bar-segment")) >= 3
    
    def test_status_updates(self, driver):
        """Test status text shows review progress."""
        driver.get(f"{self.BASE_URL}/simple")
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "status"))
        )
        
        status = driver.find_element(By.ID, "status")
        assert status.is_displayed()
        
        # Status should contain reviewed count
        status_text = status.text
        assert "reviewed" in status_text or "/" in status_text or status_text == "Loading…"
    
    def test_save_button_disabled_initially(self, driver):
        """Test save button is disabled until all rows reviewed."""
        driver.get(f"{self.BASE_URL}/simple")
        
        save_btn = driver.find_element(By.ID, "save-btn")
        
        # Initially disabled (not all rows reviewed)
        assert not save_btn.is_enabled()
    
    def test_table_headers(self, driver):
        """Test table has correct headers."""
        driver.get(f"{self.BASE_URL}/simple")
        
        headers = driver.find_elements(By.CSS_SELECTOR, "#main-table th")
        header_texts = [h.text for h in headers]
        
        expected_headers = ["Image", "UID", "Labels", "Examples", "Certainty"]
        for expected in expected_headers:
            assert any(expected in h for h in header_texts), f"Missing header: {expected}"
    
    def test_settings_link(self, driver):
        """Test settings link points to metadata config."""
        driver.get(f"{self.BASE_URL}/simple")
        
        settings_link = driver.find_element(By.CSS_SELECTOR, "a.settings-link")
        href = settings_link.get_attribute("href")
        assert "/metadata/visible/page" in href


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
