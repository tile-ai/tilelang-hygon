import subprocess
import re

# People using this function probably want to set CUDA_VISIBLE_DEVICES to the free devices,
# but CUDA reads CUDA_VISIBLE_DEVICES only at initialization time,
# so do not import torch at the top level because it initializes CUDA immediately.
def get_free_devices():
    """
    Executes the 'hy-smi' command, parses its output to find HCUs
    with 0% VRAM usage, and returns their HCU numbers.

    Returns:
        list: A list of integer HCU numbers with 0.0% VRAM usage.
              Returns an empty list if the command fails or no such HCUs are found.
    """
    try:
        # Execute the hy-smi command
        result = subprocess.run(['hy-smi'], capture_output=True, text=True, check=True)
        output = result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error executing 'hy-smi': {e}")
        return []

    free_hcus = []
    lines = output.strip().split('\n')

    header_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("HCU"):
            header_line_index = i
            break

    if header_line_index == -1:
        print("Could not find header row in hy-smi output.")
        return []

    vram_col_index = lines[header_line_index].split().index("VRAM%")
    for line in lines[header_line_index + 1:]:
        # Match lines that start with a digit (the HCU number)
        if re.match(r'^\d', line.strip()):
            parts = line.split()
            # Check for 0% VRAM usage. The value might be '0%' or '0.0%'.
            if float(parts[vram_col_index].strip('%')) == 0.0:
                free_hcus.append(int(parts[0]))

    return free_hcus
