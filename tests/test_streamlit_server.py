import subprocess
import time
import sys
import urllib.request

print("Starting Streamlit server test...")
proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "127.0.0.1"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

try:
    # Wait for server to bind port
    success = False
    for attempt in range(12):
        time.sleep(1)
        if proc.poll() is not None:
            out, err = proc.communicate()
            print("Streamlit exited early! STDOUT:", out, "STDERR:", err)
            sys.exit(1)
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2)
            if resp.status == 200:
                print(f"Streamlit server is healthy! HTTP Status: {resp.status}")
                success = True
                break
        except Exception:
            pass

    if not success:
        proc.terminate()
        out, err = proc.communicate(timeout=3)
        print("Could not reach Streamlit health endpoint within 12 seconds.")
        print("STDOUT:", out)
        print("STDERR:", err)
        sys.exit(1)
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("Streamlit process stopped cleanly.")
