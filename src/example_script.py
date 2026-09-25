try:
    import numpy as np

    arr = np.array([1,2,3])

    print(arr)

    print("Example script is working")

except ImportError:
    print("""Failed to load numpy, are you running this with uv?
try: uv run example_script.py
If that also fails, check if you have the updated uv.lock file from gitlab""")
