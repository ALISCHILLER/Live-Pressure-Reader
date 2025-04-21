import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# اطلاعات مراحل و وضعیت انجام
steps = [
    "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"
]
durations = [12, 20, 15, 10, 15, 3]
completed = [False, False, False, False, False, False]  # مراحل 3 و 6 کامل شده

# رنگ‌ها و برچسب‌ها
colors = ['skyblue' if not completed[i] else 'green' for i in range(len(completed))]
labels = [f"{steps[i]} ({'Done' if completed[i] else 'Not Done'})" for i in range(len(steps))]

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
done_patch = mpatches.Patch(color='green', label='Completed')
ax.legend(handles=[todo_patch, done_patch], loc='lower right')

plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)

# ذخیره تصویر
plt.savefig("zar_visit_supervisor_gantt_chart.png")
plt.show()
