import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

# Get the path to the .ttf file
font_path = Path(__file__).parent.parent / "v1_report" / "THSarabun.ttf"

# Create a FontProperties object
prop = fm.FontProperties(fname=font_path)

# Sample plot using the custom font
plt.figure(figsize=(6, 4))
plt.text(0.5, 0.5, "สวัสดี ภาษาไทย", fontproperties=prop, ha='center', va='center')
plt.title("กราฟตัวอย่าง", fontproperties=prop)
plt.show()
