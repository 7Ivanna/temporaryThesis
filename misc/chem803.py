import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

file = open("MockGrades_24F.txt", "r")
content=file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Distribution of grades over the years
for year in years:
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10)

axs[0].set_xlim(0, 100)

axs[0].set_title('Distribution of Grades Over the Years')
axs[0].set_xlabel('Grades')
axs[0].set_ylabel('Frequency')
axs[0].legend(title='Year')

# Mean and standard deviation over the years
axs[1].errorbar(years, means, yerr=std_devs, fmt='-o', capsize=5, label='Mean ± SD')
axs[1].set_title('Mean and Standard Deviation of Grades by Year')
axs[1].set_xlim(2014, 2024)

axs[1].set_ylabel('Grades')
axs[1].grid(True)

#  Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color='purple')
axs[2].set_title('Number of Students by Year')
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Number of Students')
axs[2].grid(True)

# Adjust layout for better readability
plt.tight_layout()

\
# Show the figure
plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Set up figure for horizontal stacking (e.g., for two-column journal fit)
fig, axs = plt.subplots(1, 3, figsize=(14, 5))  # 1 row, 3 columns

# Overall title for the figure
fig.suptitle('Analysis of Grades and Student Participation Over the Years', fontsize=16)

# Subplot 1: Distribution of grades over the years
for year in years:
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10)

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades', fontsize=12)
axs[0].set_xlabel('Grades (0-100)', fontsize=10)
axs[0].set_ylabel('Frequency', fontsize=10)
axs[0].legend(title='Year', fontsize=9)

# Subplot 2: Mean and standard deviation over the years
axs[1].errorbar(years, means, yerr=std_devs, fmt='-o', capsize=5, label='Mean ± SD')
axs[1].set_title('Mean and Standard Deviation', fontsize=12)
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=10)
axs[1].set_ylabel('Grades (0-100)', fontsize=10)
axs[1].grid(True)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color='purple')
axs[2].set_title('Number of Students', fontsize=12)
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=10)
axs[2].set_ylabel('Number of Students', fontsize=10)
axs[2].grid(True)
axs[2].legend(fontsize=9)

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF for journal submission
with PdfPages('grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Set up a creative color palette
colors = sns.color_palette("mako",len(years)) 
# Create a horizontally stacked figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})


# Subplot 1: Distribution of grades over the years
for idx, year in enumerate(years):
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10, color=colors[idx], edgecolor='black')

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades from 2015 - 2024', fontsize=14, color='darkgreen')
axs[0].set_xlabel('Grades (0-100)', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].legend(title='Year', fontsize=9, loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines

# Subplot 2: Mean and standard deviation over the years
axs[1].errorbar(years, means, yerr=std_devs, fmt='-o', capsize=5, label='Mean ± SD')
axs[1].set_title('Mean and Standard Deviation of Grades from 2015 - 2024', fontsize=12)
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=10)
axs[1].set_ylabel('Grades (0-100)', fontsize=10)
axs[1].grid(True)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color='purple')
axs[2].set_title('Number of Students from 2015 - 2024', fontsize=12)
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=10)
axs[2].set_ylabel('Number of Students', fontsize=10)
axs[2].grid(True)
axs[2].legend(fontsize=9)

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])


# Save as PDF for journal submission
with PdfPages('creative_grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Create a horizontally stacked figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})

# Set up a creative color palette
colors = sns.color_palette("mako", len(years))

# Subplot 1: Distribution of grades over the years
for idx, year in enumerate(years):
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10, color=colors[idx], edgecolor='black')

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades from 2015 - 2024', fontsize=14, color='darkgreen')
axs[0].set_xlabel('Grades (0-100)', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].legend(title='Year', fontsize=9, loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines

# Subplot 2: Mean and standard deviation using Seaborn
# Create a DataFrame for easier plotting with Seaborn
import pandas as pd

data_for_seaborn = pd.DataFrame({
    'Year': years,
    'Mean': means,
    'Standard Deviation': std_devs
})

# Use seaborn's lineplot for mean and standard deviation
sns.lineplot(data=data_for_seaborn, x='Year', y='Mean', ax=axs[1], marker='o', label='Mean', color='darkorange')
axs[1].fill_between(data_for_seaborn['Year'], 
                    data_for_seaborn['Mean'] - data_for_seaborn['Standard Deviation'], 
                    data_for_seaborn['Mean'] + data_for_seaborn['Standard Deviation'], 
                    color='darkorange', alpha=0.2, label='± 1 SD')

axs[1].set_title('Mean and Standard Deviation of Grades from 2015 - 2024', fontsize=12)
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=10)
axs[1].set_ylabel('Grades (0-100)', fontsize=10)
axs[1].grid(True)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color='purple')
axs[2].set_title('Number of Students from 2015 - 2024', fontsize=12)
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=10)
axs[2].set_ylabel('Number of Students', fontsize=10)
axs[2].grid(True)
axs[2].legend(fontsize=9)

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF for journal submission
with PdfPages('creative_grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Set a cohesive color palette
color_palette = sns.color_palette("Set2", 3)  # Use Set2 for better visibility and cohesion

# Create a horizontally stacked figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})

# Overall title with consistent font size and style
fig.suptitle('Analysis of Grades and Student Participation from 2015 - 2024', fontsize=20, fontweight='bold', color='darkblue')

# Subplot 1: Distribution of grades over the years
for idx, year in enumerate(years):
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10, color=color_palette[0], edgecolor='black', alpha=0.7)

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades', fontsize=16, color='darkgreen')
axs[0].set_xlabel('Grades (0-100)', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].legend(title='Year', fontsize=9, loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Mean and standard deviation using Seaborn
data_for_seaborn = pd.DataFrame({
    'Year': years,
    'Mean': means,
    'Standard Deviation': std_devs
})

# Plot mean with shaded standard deviation area
sns.lineplot(data=data_for_seaborn, x='Year', y='Mean', ax=axs[1], marker='o', label='Mean', color=color_palette[1])
axs[1].fill_between(data_for_seaborn['Year'], 
                    data_for_seaborn['Mean'] - data_for_seaborn['Standard Deviation'], 
                    data_for_seaborn['Mean'] + data_for_seaborn['Standard Deviation'], 
                    color=color_palette[1], alpha=0.2, label='± 1 SD')

axs[1].set_title('Mean and Standard Deviation', fontsize=16, color='darkgreen')
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=12)
axs[1].set_ylabel('Grades (0-100)', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color=color_palette[2])
axs[2].set_title('Number of Students', fontsize=16, color='darkgreen')
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=12)
axs[2].set_ylabel('Number of Students', fontsize=12)
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(fontsize=9)

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF for journal submission
with PdfPages('cohesive_grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Create a horizontally stacked figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})

# Set a cohesive color palette for all plots
colors = sns.color_palette("Set2", 3)

# Overall title with consistent font size and style
fig.suptitle('Analysis of Grades and Student Participation from 2015 - 2024', fontsize=20, fontweight='bold', color='darkblue')

# Subplot 1: Distribution of grades over the years
for idx, year in enumerate(years):
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10, color=colors[0], edgecolor='black', alpha=0.7)

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades', fontsize=16, color='darkgreen')
axs[0].set_xlabel('Grades (0-100)', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].legend(title='Year', fontsize=9, loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Mean and standard deviation using Seaborn
data_for_seaborn = pd.DataFrame({
    'Year': years,
    'Mean': means,
    'Standard Deviation': std_devs
})

# Plot mean with shaded standard deviation area
sns.lineplot(data=data_for_seaborn, x='Year', y='Mean', ax=axs[1], marker='o', label='Mean', color=colors[1])
axs[1].fill_between(data_for_seaborn['Year'], 
                    data_for_seaborn['Mean'] - data_for_seaborn['Standard Deviation'], 
                    data_for_seaborn['Mean'] + data_for_seaborn['Standard Deviation'], 
                    color=colors[1], alpha=0.2, label='± 1 SD')

axs[1].set_title('Mean and Standard Deviation', fontsize=16, color='darkgreen')
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=12)
axs[1].set_ylabel('Grades (0-100)', fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color=colors[2])
axs[2].set_title('Number of Students', fontsize=16, color='darkgreen')
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=12)
axs[2].set_ylabel('Number of Students', fontsize=12)
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(fontsize=9)

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF for journal submission
with PdfPages('cohesive_grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file = open("MockGrades_24F.txt", "r")
content = file.readlines()
file.close()
grades_data = {int(l.split(':')[0]): [int(j) for j in l.split(':')[1].strip().split('\t')] for l in content}

years = sorted(grades_data.keys())
means = [np.mean(grades_data[year]) for year in years]
std_devs = [np.std(grades_data[year]) for year in years]
num_students = [len(grades_data[year]) for year in years]

# Create a horizontally stacked figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})

# Set up a creative color palette
colors = sns.color_palette("mako", len(years))

# Subplot 1: Distribution of grades over the years
for idx, year in enumerate(years):
    sns.histplot(grades_data[year], ax=axs[0], label=str(year), binwidth=10, color=colors[idx], edgecolor='black')

axs[0].set_xlim(0, 100)
axs[0].set_title('Distribution of Grades from 2015 - 2023', fontsize=14, color='black')
axs[0].set_xlabel('Grades (0-100)', fontsize=12)
axs[0].set_ylabel('Frequency', fontsize=12)
axs[0].legend(title='Year', fontsize=9, loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.6)  # Subtle gridlines

# Subplot 2: Mean and standard deviation using Seaborn
data_for_seaborn = pd.DataFrame({
    'Year': years,
    'Mean': means,
    'Standard Deviation': std_devs
})

sns.lineplot(data=data_for_seaborn, x='Year', y='Mean', ax=axs[1], marker='o', label='Mean', color='darkorange')
axs[1].fill_between(data_for_seaborn['Year'], 
                    data_for_seaborn['Mean'] - data_for_seaborn['Standard Deviation'], 
                    data_for_seaborn['Mean'] + data_for_seaborn['Standard Deviation'], 
                    color='darkorange', alpha=0.2, label='± 1 SD')

axs[1].set_title('Mean and Standard Deviation of Grades from 2015 - 2023', fontsize=12)
axs[1].set_xlim(2014, 2024)
axs[1].set_xlabel('Year', fontsize=10)
axs[1].set_ylabel('Grades (0-100)', fontsize=10)
axs[1].grid(True)
axs[1].legend(fontsize=9)

# Subplot 3: Number of students over the years
axs[2].plot(years, num_students, '-o', label='Number of Students', color='purple')
axs[2].set_title('Number of Students from 2015 - 2023', fontsize=12)
axs[2].set_xlim(2014, 2024)
axs[2].set_xlabel('Year', fontsize=10)
axs[2].set_ylabel('Number of Students', fontsize=10)
axs[2].grid(True)
axs[2].legend(fontsize=9)

# Add labels (a), (b), and (c) beside the subplots
axs[0].text(-0.15, 1.05, 'a)', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
axs[1].text(-0.15, 1.05, 'b)', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
axs[2].text(-0.15, 1.05, 'c)', transform=axs[2].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

# Adjust layout for better readability, leaving space for the overall title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save as PDF for journal submission
with PdfPages('creative_grades_distribution_horizontal.pdf') as pdf:
    pdf.savefig(fig)

# Show the figure
plt.show()

# %%
