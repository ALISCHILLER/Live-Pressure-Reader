import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# اطلاعات مراحل و وضعیت انجام
steps = [
    "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
]
durations = [20, 30, 20, 35, 30, 10]
completed = [False] * len(steps)  # همه مراحل هنوز تکمیل نشده‌اند

# رنگ‌ها و برچسب‌ها
colors = ['skyblue' for _ in completed]
labels = [f"{steps[i]} ({'Not Done'})" for i in range(len(steps))]

# ترسیم نمودار Gantt
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = list(range(len(labels)))
ax.barh(y_pos, durations, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Duration (Days)')
ax.set_title('Gantt Chart – App Zar Visit Supervisor')

# راهنمای رنگ‌ها
todo_patch = mpatches.Patch(color='skyblue', label='Not Completed')
ax.legend(handles=[todo_patch], loc='lower right')

plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)

# ذخیره تصویر
plt.savefig("zar_visit_supervisor_gantt_chart.png")
plt.show()
