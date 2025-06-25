import numpy as np
import matplotlib.pyplot as plt
from textwrap import dedent

def layout2coord(layout, use_dedent=True):
    """
    Converts a layout string into coordinates for source and target cells.
    
    In the layout string:
      - 'S' marks a source cell,
      - 'T' marks a target cell,
      - ' ', '.', or '_' mark empty cells.
      
    The grid is centered and the top row is given the highest y value.
    
    Any tabs in the layout are converted to spaces.
    """
    # Convert tabs to spaces (4 spaces per tab, adjust as needed)
    layout = layout.expandtabs(4)
    
    # Optionally dedent the layout (this removes common indentation)
    if use_dedent:
        layout = dedent(layout).strip('\n')
    else:
        layout = layout.rstrip('\n')
    
    # Split into lines and pad them to the same length.
    lines = layout.splitlines()
    max_length = max(len(line) for line in lines)
    grid_chars = [list(line.ljust(max_length)) for line in lines]
    grid_array = np.array(grid_chars)
    
    # Grid dimensions: N rows, M columns.
    N, M = grid_array.shape
    
    # Compute centered coordinates.
    # x increases to the right.
    x = np.arange(M) - (M - 1) / 2.0
    # y decreases from top to bottom (top row gets highest y)
    y = (N - 1) / 2.0 - np.arange(N)
    
    # Create a meshgrid.
    X, Y = np.meshgrid(x, y)
    
    # Create boolean masks for source and target cells.
    s_mask = (grid_array == 'S')
    t_mask = (grid_array == 'T')
    
    # Extract coordinates for source and target cells.
    sX = X[s_mask]
    sY = Y[s_mask]
    tX = X[t_mask]
    tY = Y[t_mask]
    
    return sX, sY, tX, tY

def plot_root_layout(layout):
    sX, sY, tX, tY = layout2coord(layout)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sX, sY, marker='s', s=50, color='red')
    ax.scatter(tX, tY, marker='s', s=50, color='green')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Root Coordinates')
    plt.savefig("layout.png", dpi=100, bbox_inches='tight')
    plt.close()
  

if __name__ == '__main__':
    # Define the layout as a multi-line string.
    layout = """
           ST
        S  ST    S
        S  ST    S
        S  ST    S
        T    
    """

    # Print the textual layout so you see it in the console.
    print("Textual layout:")
    print(dedent(layout).strip())

    print(layout2coord(layout))
    plot_root_layout(layout)

