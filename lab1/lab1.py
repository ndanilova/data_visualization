import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from scipy.stats import chi2_contingency


path = "monster_com_job_sample.csv"
df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

print(df.head(5))

missing_stats = pd.DataFrame({
    'Total cells': df.shape[0],
    'Empty cells': df.isnull().sum()
})
missing_stats['Percentage of passes (%)'] = (missing_stats['Empty cells'] / df.shape[0] * 100).round(2)
print(missing_stats)


threshold = 0.75 * df.shape[1]

empty_rows_mask = df.isnull().sum(axis=1) > threshold

empty_rows_count = empty_rows_mask.sum()

print(f'Количество строк с более чем 75% пропусков: {empty_rows_count}')


missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_percent = (missing_data / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    "column": missing_percent.index,
    "percent": missing_percent.values
})

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
ax = sns.barplot(
    data=missing_df,
    x="percent",
    y="column",
    hue="column",
    palette="flare",
    dodge=False,
    legend=False
)

for i, v in enumerate(missing_df["percent"]):
    ax.text(v + 0.5, i, f"{v}%", color='black', va='center', fontweight='bold')

plt.title("Percentage of skips by column", fontsize=16)
plt.xlabel("Percentage of skips (%)", fontsize=12)
plt.ylabel("Columns", fontsize=12)
plt.tight_layout()
plt.show()


initial_shape = df.shape

df_dropped = df.dropna()

final_shape = df_dropped.shape

removed_rows = initial_shape[0] - final_shape[0]

print(f"Размер до удаления: {initial_shape}")
print(f"Размер после удаления: {final_shape}")
print(f"Удалено строк: {removed_rows}")
print(f"Процент удалённых строк: {removed_rows / initial_shape[0] * 100:.2f}%")


single_value_columns = df.columns[df.nunique() == 1].tolist()
print("Столбцы с одним уникальным значением:", single_value_columns)

value_counts = defaultdict(int)
for column in df.columns:
    for value in df[column].dropna().astype(str):
        value_counts[value] += 1

common_values = {k: v for k, v in value_counts.items() if v > 1}
print("Значения в нескольких столбцах:", common_values)

print("Типы данных:")
print(df.dtypes)


is_uniq_id_unique = df['uniq_id'].is_unique
print(f"Все значения в 'uniq_id' уникальны: {is_uniq_id_unique}")

duplicates_in_uniq_id = df['uniq_id'].duplicated().sum()
print(f"Количество дубликатов в 'uniq_id': {duplicates_in_uniq_id}")

unique_columns = []
for column in df.columns:
    if column != 'uniq_id' and df[column].nunique() == len(df):
        unique_columns.append(column)

print(f"Столбцы, где все значения уникальны: {unique_columns}")

columns_to_check = [col for col in df.columns if col not in ['uniq_id', 'page_url']]

duplicates = df[df.duplicated(subset=columns_to_check, keep=False)]

duplicates_sorted = duplicates.sort_values(by=columns_to_check)

if not duplicates_sorted.empty:
    print(f"Найдено дубликатов: {len(duplicates_sorted)}")
    print("\nПримеры дубликатов (первые 10 записей):")
    print(duplicates_sorted[columns_to_check].head(10))
else:
    print("Дубликаты не найдены.")


locations = [
    "Madison, WI 53702",
    "Houston, TX",
    "90210",
    "CA",
    "New York, NY 10001",
    "Chicago IL"
]

patterns = {
    "city_state_zip": r"^([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5})$",  # "Город, ШТАТ ИНДЕКС"
    "city_state": r"^([A-Za-z\s]+),\s*([A-Z]{2})$",                 # "Город, ШТАТ"
    "state_only": r"^[A-Z]{2}$",                                    # "ШТАТ" (только код)
    "zip_only": r"^\d{5}$",                                         # "ИНДЕКС"
    "city_only": r"^[A-Za-z\s]+$"                                   # "Город"
}

def detect_format(loc):
    if pd.isna(loc) or loc.strip() == "":
        return "unknown"
    for fmt, pattern in patterns.items():
        if re.fullmatch(pattern, loc.strip()):
            return fmt
    return "unknown"

df["location_format"] = df["location"].apply(detect_format)


format_counts = df["location_format"].value_counts()

plt.figure(figsize=(10, 6))
format_counts.plot(kind="bar", color="skyblue")
plt.title("Distribution of formats in the location column")
plt.xlabel("Format")
plt.ylabel("Number of entries")
plt.xticks(rotation=45)
plt.show()


df_parsed = df.copy()

def parse_location(location):
    if pd.isna(location):
        return None, None, None

    location = str(location).strip()

    # Удаляем явно некорректные строки
    if any(phrase in location for phrase in ["DePuy Synthes", "Contact name", "Sr. Process Engineer"]):
        return None, None, None

    city = None
    state = None
    zip_code = None

    state_pattern = r'\b([A-Z]{2})\b'
    zip_pattern = r'\b(\d{5})\b'

    state_match = re.search(state_pattern, location)
    if state_match:
        state = state_match.group(1)
        location = location.replace(state, '').strip()

    zip_match = re.search(zip_pattern, location)
    if zip_match:
        zip_code = zip_match.group(1)
        location = location.replace(zip_code, '').strip()

    if location:
        city = re.sub(r'^[,\s;]+|[,\s;]+$', '', location)
        if not city:  # Если после очистки ничего не осталось
            city = None

    return city, state, zip_code

df_parsed[['city', 'state', 'zip']] = df_parsed['location'].apply(
    lambda x: pd.Series(parse_location(x)))

# Преобразуем zip код в числовой формат
df_parsed['zip'] = pd.to_numeric(df_parsed['zip'], errors='coerce').astype('Int64')

# Проверяем результаты
print(df_parsed[['location', 'city', 'state', 'zip']].head(10))

failed_rows = df[(df_parsed['city'].isna()) & (df_parsed['state'].isna()) & (df_parsed['zip'].isna())].copy()

def advanced_location_parser(loc):
    company_match = re.search(r'(?:at|in|from)\s+([^,]+,\s*[A-Z]{2}\s*\d{5})', loc, flags=re.IGNORECASE)
    if company_match:
      return advanced_location_parser(company_match.group(1))

    if pd.isna(loc) or not isinstance(loc, str):
        return None, None, None

    loc = loc.strip()

    # Отсеиваем явно некорректные строки
    if any(x in loc.lower() for x in ['contact name', 'recruiter', 'phone', 'fax', 'hr director', 'not given']):
        return None, None, None

    # Пытаемся извлечь адрес из строк с контактной информацией
    address_match = re.search(r'Address\s*([^\n]+)', loc, flags=re.IGNORECASE)
    if address_match:
        loc = address_match.group(1).strip()

    # Исправляем слипшиеся названия городов (например: "AvePortland" -> "Ave, Portland")
    if not re.search(r'[a-zA-Z],\s*[A-Z]{2}', loc) and re.search(r'[a-z][A-Z]', loc):
        loc = re.sub(r'([a-z])([A-Z][a-z])', r'\1, \2', loc)

    # Стандартизация разделителей
    loc = re.sub(r'\s{2,}', ' ', loc)
    loc = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', loc)
    loc = re.sub(r'([a-z])([A-Z])', r'\1 \2', loc)
    loc = re.sub(r'(\bAve|\bSt|\bRd|\bDr|\bBlvd)([A-Z][a-z])', r'\1, \2', loc)

    city, state, zip_code = None, None, None

    zip_match = re.search(r'\b(\d{5})\b', loc)
    if zip_match:
        zip_code = zip_match.group(1)
        loc = loc.replace(zip_code, '').strip()

    state_match = re.search(r'\b([A-Z]{2})\b', loc)
    if state_match:
        state = state_match.group(1)
        loc = loc.replace(state, '').strip()

    if loc:
        city = re.sub(r'^[,\s;]+|[,\s;]+$', '', loc)
        city = None if city == '' else city.title()

    return city, state, zip_code

df[['city', 'state', 'zip']] = df['location'].apply(
    lambda x: pd.Series(advanced_location_parser(x)))

# Преобразуем ZIP-код в числовой формат
df['zip'] = pd.to_numeric(df['zip'], errors='coerce').astype('Int64')

# Анализ результатов
failed_rows = df[df['city'].isna() & df['state'].isna() & df['zip'].isna()]
print(f"Успешно распарсено: {len(df) - len(failed_rows)}/{len(df)} строк")
print("\nПримеры неудачных строк:")
print(failed_rows['location'].head(20).to_string())

patterns_salary = {
    "range_hour": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*/\s*hour\s*$",
    "range_day": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*/\s*day\s*$",
    "range_week": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*/\s*week\s*$",
    "range_month": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*/\s*month\s*$",
    "range_year": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*/\s*year\s*$",
    "single_hour": r"^\s*\$?\s*[\d,]+\.\d+\s*/\s*hour\s*$",
    "single_day": r"^\s*\$?\s*[\d,]+\.\d+\s*/\s*day\s*$",
    "single_week": r"^\s*\$?\s*[\d,]+\.\d+\s*/\s*week\s*$",
    "single_month": r"^\s*\$?\s*[\d,]+\.\d+\s*/\s*month\s*$",
    "single_year": r"^\s*\$?\s*[\d,]+\.\d+\s*/\s*year\s*$",
    "from_hour_plus": r"^\s*\$?\s*[\d,]+\.\d+\+?\s*/\s*hour\s*$",
    "from_year_plus": r"^\s*\$?\s*[\d,]+\.\d+\+?\s*/\s*year\s*$",
    "from_month_plus": r"^\s*\$?\s*[\d,]+\.\d+\+?\s*/\s*month\s*$",
    "upto": r"^\s*up to\s+\$?\s*[\d,]+\.\d+\s*$",
    "range_no_period": r"^\s*\$?\s*[\d,]+\.\d+\s*-\s*\$?\s*[\d,]+\.\d+\s*\$?\s*$",
    "single_no_period": r"^\s*\$?\s*[\d,]+\.\d+\s*$"
}

def detect_salary_format(sal):
    if pd.isna(sal) or str(sal).strip() == "":
        return "unknown"
    sal = str(sal).strip().lower()
    for fmt, pattern in patterns_salary.items():
        if re.fullmatch(pattern, sal, re.IGNORECASE):
            return fmt
    return "unknown"

df["salary_format"] = df["salary"].apply(detect_salary_format)

plt.figure(figsize=(14, 7))
counts = df["salary_format"].value_counts()

# Группируем похожие форматы для лучшей читаемости
grouped_counts = pd.Series({
    "hourly": counts[counts.index.str.contains('hour')].sum(),
    "daily": counts[counts.index.str.contains('day')].sum(),
    "weekly": counts[counts.index.str.contains('week')].sum(),
    "monthly": counts[counts.index.str.contains('month')].sum(),
    "yearly": counts[counts.index.str.contains('year')].sum(),
    "no_period": counts[counts.index.str.contains('no_period')].sum(),
    "upto": counts["upto"],
    "unknown": counts["unknown"] if "unknown" in counts else 0
})

# Сортируем для удобства
grouped_counts = grouped_counts.sort_values(ascending=False)

bars = plt.bar(grouped_counts.index, grouped_counts.values, color='skyblue')
plt.title('Distribution of salary formats', pad=20)
plt.xlabel('Format type')
plt.ylabel('Number of entries')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Детальная статистика по форматам:")
print(counts.sort_values(ascending=False))

def extract_salary_values(salary):
    if pd.isna(salary) or str(salary).strip() == "":
        return None, None, None

    salary_str = str(salary).strip()
    amounts = re.findall(r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', salary_str)
    clean_amounts = [float(amt.replace(',', '')) for amt in amounts if amt.replace(',', '').replace('.', '').isdigit()]

    if not clean_amounts:
        return None, None, None

    min_salary = min(clean_amounts)
    max_salary = max(clean_amounts) if len(clean_amounts) > 1 else None

    period_patterns = {
        'hour': r'/hour|/hr|\bper hour\b',
        'day': r'/day|\bper day\b',
        'week': r'/week|/wk|\bper week\b',
        'month': r'/month|/mo|\bper month\b',
        'year': r'/year|/yr|\bper year\b'
    }

    period = next((p for p, pat in period_patterns.items() if re.search(pat, salary_str, re.I)), None)

    if 'up to' in salary_str.lower() or 'до' in salary_str.lower():
        min_salary, max_salary = None, max(clean_amounts)
    elif '+' in salary_str:
        max_salary = None

    return min_salary, max_salary, period

df[['min_salary', 'max_salary', 'period']] = df['salary'].apply(
    lambda x: pd.Series(extract_salary_values(x)))
df['min_salary'] = pd.to_numeric(df['min_salary'], errors='coerce')
df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')

# Анализ по периодам
period_stats = df[df['period'].notna()].groupby('period')['min_salary'].agg(['mean', 'median', 'count'])
period_stats = period_stats.sort_values('count', ascending=False)

plt.figure(figsize=(12, 6))

colors = {'mean': '#1f77b4', 'median': '#d62728'}  # Синий и красный
y_pos = np.arange(len(period_stats))
bar_height = 0.35

# Рисуем горизонтальные столбцы
mean_bars = plt.barh(y_pos - bar_height/2, period_stats['mean'],
                    height=bar_height, color=colors['mean'], label='Mean')
median_bars = plt.barh(y_pos + bar_height/2, period_stats['median'],
                      height=bar_height, color=colors['median'], label='Median')

plt.title('Comparison of the average and median minimum wages by period',
          pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Salary ($)', fontsize=12)
plt.ylabel('Payment period', fontsize=12)
plt.yticks(y_pos, period_stats.index)
plt.grid(axis='x', linestyle='--', alpha=0.4)

def currency_formatter(x, pos):
    return f'${x:,.0f}'
plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))

for bars in [mean_bars, median_bars]:
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(period_stats['mean']) * 0.02, bar.get_y() + bar.get_height()/2,
                f'${width:,.0f}',
                va='center', ha='left', fontsize=10)

plt.legend(framealpha=1, loc='upper right')

plt.gcf().set_facecolor('#f5f5f5')
plt.gca().set_facecolor('#ffffff')
plt.xlim(0, max(period_stats['mean']) * 1.2)

plt.tight_layout()
plt.show()

print("Статистика по периодам:")
print(period_stats)

known_periods = df[df['min_salary'].notna() & df['period'].notna()]

periods = known_periods['period'].unique()

sns.set(style='whitegrid', palette='pastel')

color = "#69b3a2"

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20, 12))
axs = axs.flatten()

for i, period in enumerate(periods):
    ax = axs[i]
    period_data = known_periods[known_periods['period'] == period]['min_salary'].dropna()

    # Гистограмма
    ax.hist(period_data, bins=20, density=True, alpha=0.7, color=color, edgecolor='gray')

    if len(period_data) > 1 and np.std(period_data) > 0:
        mu, std = norm.fit(period_data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'black', linewidth=2)
        ax.set_title(f'Salary distribution ({period})\nμ = ${mu:,.0f}, σ = ${std:,.0f}',
                     fontsize=12, fontweight='semibold')
    else:
        ax.set_title(f'Salary distribution ({period})\nInsufficient data',
                     fontsize=12, fontweight='semibold')

    ax.set_xlabel('Salary ($)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='both', which='major', labelsize=10)

# Удалим пустые оси
for j in range(len(periods), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(pad=4)
plt.show()



unknown_periods = df[df['period'].isna() & df['min_salary'].notna()].copy()

# 3. Определяем ближайший период для зарплат без указания периода
period_stats = known_periods.groupby('period')['min_salary'].agg(['mean', 'std', 'count'])

def find_closest_period(salary):
    if pd.isna(salary):
        return None

    min_distance = float('inf')
    closest_period = None

    for period, stats in period_stats.iterrows():
        if pd.notna(stats['mean']) and stats['std'] > 0:
            distance = abs((salary - stats['mean']) / stats['std'])

            if distance < min_distance:
                min_distance = distance
                closest_period = period
            elif distance == min_distance:
                closest_period = None  # неоднозначно

    return closest_period

# Применяем предсказание
unknown_periods['predicted_period'] = unknown_periods['min_salary'].apply(find_closest_period)

# Визуализация результатов
if not unknown_periods.empty:
    plt.figure(figsize=(10, 6))
    prediction_counts = unknown_periods['predicted_period'].value_counts()
    prediction_counts = prediction_counts[prediction_counts.index.notna()]

    if not prediction_counts.empty:
        prediction_counts.plot(kind='bar', color='lightgreen')
        plt.title('Distribution of predicted pay periods for salaries without specifying the period')
        plt.xlabel('Period')
        plt.ylabel('Number of entries')

        for i, v in enumerate(prediction_counts):
            plt.text(i, v + 0.5, str(v), ha='center')

        plt.show()
    else:
        print("Не удалось определить период для всех записей")

# Объединяем известные и предсказанные периоды
df['final_period'] = df['period'].copy()
df.loc[unknown_periods.index, 'final_period'] = unknown_periods['predicted_period']


# Выводим статистику
print("\nСтатистика по известным периодам:")
print(period_stats)

print("\nРезультаты предсказания периодов:")
print(f"Всего записей без периода: {len(unknown_periods)}")
print(f"Удалось определить период для: {unknown_periods['predicted_period'].notna().sum()} записей")

print("\nПримеры предсказаний:")
sample = unknown_periods[unknown_periods['predicted_period'].notna()].sample(5, random_state=42)
for idx, row in sample.iterrows():
    print(f"Зарплата: ${row['min_salary']:,.2f} → Предсказанный период: {row['predicted_period']}")


def standardize_job_type(job_type):

    if pd.isna(job_type):
        return np.nan

    job = str(job_type).strip().lower()

    # Удаляем ненужные префиксы
    job = job.replace('job type', '').replace('employee', '').strip()

    # Удаляем лишние запятые и разделители
    job = job.split(',')[0].split('/')[0].strip()

    # Определяем основной тип
    if 'part' in job:
        return 'Part-Time'
    elif 'full' in job:
        return 'Full-Time'
    elif any(x in job for x in ['temp', 'contract', 'project']):
        return 'Contractor'
    else:
        return 'Other'

df['job_type'] = df['job_type'].apply(standardize_job_type)

print(df['job_type'].value_counts(dropna=False))

data = {'job_type': [
    'Full Time',
    'Part Time Employee',
    'Full-Time, Employee',
    'Per Diem',
    'Intern',
    'Employee',
    'PT',
    'FT Employee',
    'Full Time, Temporary',
    'Per Diem Intern',
    'Contractor',
    'Temporary',
    None  # Проверяем обработку пропусков
]}

df = pd.DataFrame(data)

def split_job_type(job_type):
    if pd.isna(job_type):
        return pd.Series(['Full Time', 'Employee'])

    original = str(job_type)
    job_lower = original.lower()

    # Определяем полноту занятости
    if re.search(r'full|ft|full[\s-]*time', job_lower):
        fullness = 'Full Time'
    elif re.search(r'part|pt|part[\s-]*time', job_lower):
        fullness = 'Part Time'
    elif re.search(r'per[\s-]*diem|daily', job_lower):
        fullness = 'Per Diem'
    else:
        fullness = 'Full Time'

    # Определяем тип занятости
    if re.search(r'employee|empl|emp', job_lower):
        job_type_category = 'Employee'
    elif re.search(r'intern|trainee', job_lower):
        job_type_category = 'Intern'
    elif re.search(r'temp|temporary|contract', job_lower):
        job_type_category = 'Contractor'
    else:
        job_type_category = 'Employee'

    return pd.Series([fullness, job_type_category])

df[['employment_fullness', 'employment_type']] = df['job_type'].apply(split_job_type)

print("Примеры преобразования:")
sample = df.head(10).copy()
sample['transform'] = sample['job_type'].astype(str) + ' → ' + \
                     sample['employment_fullness'] + ' + ' + \
                     sample['employment_type']
print(sample[['job_type', 'transform']].to_string(index=False))

plt.figure(figsize=(12, 6))
cross_tab = pd.crosstab(df['employment_fullness'], df['employment_type'])
cross_tab.plot(kind='bar', stacked=True, colormap='viridis')

plt.title('Distribution of types of employment according to the fullness of employment', pad=20)
plt.ylabel('Number', labelpad=10)
plt.xlabel('Full employment', labelpad=10)
plt.xticks(rotation=0)
plt.legend(title='Type of employment', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()


df_parsed['salary_missing'] = df_parsed['salary'].isna().astype(int)

if 'state' in df_parsed.columns:
    state_data = df_parsed.dropna(subset=['state'])

    # Выбираем топ-10 штатов
    top_states = state_data['state'].value_counts().nlargest(10).index

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='state',
        y='salary_missing',
        data=state_data[state_data['state'].isin(top_states)],
        estimator=lambda x: np.mean(x) * 100, # Процент пропусков
        errorbar=None

    )
    plt.title('Percentage of salary skips by state (Top 10)')
    plt.ylabel('Percentage of passes (%)')
    plt.xlabel('State')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 10),
                   textcoords='offset points')

    plt.show()

    # Статистический тест
    contingency_table = pd.crosstab(state_data['salary_missing'], state_data['state'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nСтатистическая значимость зависимости от штата: p-value = {p:.4f}")
else:
    print("\nСтолбец 'state' отсутствует - анализ по штатам невозможен")

if 'employment_type' in df_parsed.columns:
    emp_data = df_parsed.dropna(subset=['employment_type'])

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x='employment_type',
        y='salary_missing',
        data=emp_data,
        estimator=lambda x: np.mean(x) * 100  # Процент пропусков
    )
    plt.title('Процент пропусков зарплаты по типам занятости')
    plt.ylabel('Процент пропусков (%)')
    plt.xlabel('Тип занятости')
    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 10),
                   textcoords='offset points')

    plt.show()

# 4. Дополнительные проверки
print("\nОбщая статистика пропусков зарплаты:")
print(f"Всего записей: {len(df_parsed)}")
print(f"Пропусков зарплаты: {df_parsed['salary_missing'].sum()} ({df_parsed['salary_missing'].mean()*100:.1f}%)")



# 3. Анализ по организациям (organization) - топ-15
if 'organization' in df_parsed.columns:
    org_data = df_parsed.dropna(subset=['organization']).copy()
    org_data['salary_missing'] = org_data['salary'].isna()  # <-- ВАЖНО!

    # Выбираем топ-15 организаций по количеству вакансий
    top_orgs = org_data['organization'].value_counts().nlargest(15).index

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(
        x='organization',
        y='salary_missing',
        data=org_data[org_data['organization'].isin(top_orgs)],
        estimator=lambda x: np.mean(x) * 100,
        errorbar=None,
        order=top_orgs
    )
    plt.title('Percentage of salary skips by organization (Top 15)', pad=20)
    plt.ylabel('Percentage of missing salaries (%)', labelpad=10)
    plt.xlabel('Organization', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 10),
                   textcoords='offset points')

    plt.tight_layout()
    plt.show()

    # Статистический тест
    contingency_table = pd.crosstab(org_data['salary_missing'], org_data['organization'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nStatistical significance for organizations: p-value = {p:.4f}")
else:
    print("\nColumn 'organization' is missing - organization analysis is not possible")

# 4. Анализ по секторам (sector)
if 'sector' in df.columns:
    sector_data = df.dropna(subset=['sector']).copy()
    sector_data['salary_missing'] = sector_data['salary'].isna()  # <-- ВАЖНО!

    top_sectors = sector_data['sector'].value_counts().nlargest(15).index

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(
        x='sector',
        y='salary_missing',
        data=sector_data[sector_data['sector'].isin(top_sectors)],
        estimator=lambda x: np.mean(x) * 100,
        errorbar=None,
        order=top_sectors
    )
    plt.title('Percentage of salary skips by sector (Top 15)', pad=20)
    plt.ylabel('Percentage of missing salaries (%)', labelpad=10)
    plt.xlabel('Sector', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 10),
                   textcoords='offset points')

    plt.tight_layout()
    plt.show()

    contingency_table = pd.crosstab(sector_data['salary_missing'], sector_data['sector'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nStatistical significance for sectors: p-value = {p:.4f}")
else:
    print("\nColumn 'sector' is missing - sector analysis is not possible")


def split_job_type(job_type):
    if pd.isna(job_type):
        return pd.Series(['Full Time', 'Employee'])

    job_lower = str(job_type).lower()

    if re.search(r'full|ft|full[\s-]*time', job_lower):
        fullness = 'Full Time'
    elif re.search(r'part|pt|part[\s-]*time', job_lower):
        fullness = 'Part Time'
    elif re.search(r'per[\s-]*diem|daily', job_lower):
        fullness = 'Per Diem'
    else:
        fullness = 'Full Time'

    if re.search(r'employee|empl|emp', job_lower):
        job_type_category = 'Employee'
    elif re.search(r'intern|trainee', job_lower):
        job_type_category = 'Intern'
    elif re.search(r'temp|temporary|contract', job_lower):
        job_type_category = 'Contractor'
    else:
        job_type_category = 'Employee'

    return pd.Series([fullness, job_type_category])

# --- Преобразование данных ---
df[['employment_fullness', 'employment_type']] = df['job_type'].apply(split_job_type)
df_parsed['salary_missing'] = df_parsed['salary'].isna()

# --- Фильтрация секторов, где есть Intern ---
sectors_with_intern = df[df['employment_type'] == 'Intern']['sector'].dropna().unique()

# --- Создание сводной таблицы только для этих секторов ---
filtered_df = df[df['sector'].isin(sectors_with_intern)]

pivot_table = filtered_df.pivot_table(
    index='sector',
    columns='employment_type',
    values='salary_missing',
    aggfunc='mean'
) * 100

# --- Построение графика ---
plt.figure(figsize=(16, 12), constrained_layout=True)
sns.heatmap(
    pivot_table,
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={'label': '% Пропущенных зарплат'},
    annot_kws={'size': 9}
)
plt.title('Процент пропусков зарплаты по секторам и типам занятости (только где есть Intern)', fontsize=16, pad=20)
plt.xlabel('Тип занятости', fontsize=12, labelpad=10)
plt.ylabel('Сектор', fontsize=12, labelpad=10)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.show()

