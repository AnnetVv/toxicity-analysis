
import os, re, sys, random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ---------- Налаштування ----------
INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
OUT_DIR = "results"
ANONYMIZE = True
LABEL = "Політик"
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)


def out(name):
    """Шлях до файлу результату."""
    return os.path.join(OUT_DIR, name)

# Парсинг файлу
def parse_text_file(filepath, label=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Назва автора: або анонімна мітка, або назва файлу
    data = {'author': label or re.sub(r'\.txt$', '', os.path.basename(filepath))}

    # Основні показники
    patterns = {k: re.search(p, content) for k, p in {
        'total_words': r'Кількість слів тексту\s*\n\s*(\d+)',
        'sentences': r'Кількість речень\s*\n\s*(\d+)',
        'vocab_size': r'Обсяг словника слів\s*\n\s*(\d+)',
        'manipulation': r'Маніпуляція\s*\n\s*(\d+)',
        'toxicity_score': r'Ступінь токсичності тексту\s*\n\s*([\d.]+)'
    }.items()}

    data.update({k: float(v.group(1)) if v else 0 for k, v in patterns.items()})

    # Емоції
    emotions = ['Здивування', 'Радість', 'Сум', 'Довіра', 'Негатив',
                'Позитив', 'Очікування', 'Гнів', 'Страх', 'Огида']

    for e in emotions:
        match = re.search(rf'{e}\s*\n\s*(\d+)', content)
        value = float(match.group(1)) if match else 0
        data[e.lower()] = value

    # Маніпулятивні тактики та токсичні категорії
    data['manipulation_tactics'] = {}
    data['toxic_categories'] = {}

    for section, key in [(r'МАНІПУЛЯЦІЇ ТЕКСТУ\n(.*?)(?=\n\n|\n[A-ZА-Я]|$)', 'manipulation_tactics'),
                         (r'Класифікація 2 для словосполучень\n(.*?)(?=\n\n|\n[A-ZА-Я]|$)', 'toxic_categories')]:
        sec = re.search(section, content, re.DOTALL)
        if sec:
            for line in sec.group(1).split('\n'):
                if '\t' in line and re.search(r'\d+$', line):
                    parts = line.strip().split('\t')
                    data[key][parts[0].strip()] = int(re.search(r'(\d+)$', parts[-1]).group(1))

    # Емоційна інтенсивність (сума всіх емоцій)
    data['emotional_intensity'] = sum(data[e.lower()] for e in emotions)

    # Співвідношення позитив/негатив
    data['positive_negative_ratio'] = (data['позитив'] / data['негатив']) if data['негатив'] > 0 else float('inf')

    return data


def load_files(folder=INPUT_DIR):
    """Читає всі .txt з папки. Прізвища з назв файлів у результати не потрапляють."""
    paths = sorted(Path(folder).glob("*.txt"))
    if not paths:
        raise SystemExit(f"У папці {folder} немає .txt-файлів")

    if ANONYMIZE:
        random.Random(42).shuffle(paths)

    data = []
    print(f"\nЧитання файлів з папки {folder}:")
    for i, p in enumerate(paths, 1):
        label = f"{LABEL} {i}" if ANONYMIZE else None
        try:
            data.append(parse_text_file(str(p), label))
            print(f"  ✓ {label or p.name}")   
        except Exception as e:
            print(f"  ✗ {label or p.name} - помилка: {e}")

    print(f"\nЗавантажено {len(data)} з {len(paths)} файлів")
    return pd.DataFrame(data)


# Графіки
def plot_bar(df, col, title, ylabel, color, filename, round_digits=1, ylim=None):
    """Стовпчикова діаграма"""
    df_sorted = df.sort_values(col, ascending=False)
    plt.figure(figsize=(16, 7))
    bars = plt.bar(df_sorted['author'], df_sorted[col], color=color, edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if ylim:
        plt.ylim(ylim)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + (0.02 if ylim else max(0.5, height * 0.02)),
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out(filename), dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_emotion_profiles(df):
    """Профілі емоцій для всіх авторів"""
    if df.empty:
        return

    emotions = ['позитив', 'радість', 'довіра', 'очікування',
                'негатив', 'гнів', 'страх', 'сум', 'огида', 'здивування']
    labels = ['Позитив', 'Радість', 'Довіра', 'Очікування',
              'Негатив', 'Гнів', 'Страх', 'Сум', 'Огида', 'Здивування']

    df_sorted = df.sort_values('toxicity_score', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    rows, cols = (n + 3) // 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        vals = [row[e] for e in emotions]
        max_val = max(vals) if max(vals) > 0 else 1
        norm_vals = [v / max_val * 100 for v in vals]

        bars = ax.bar(range(len(labels)), norm_vals, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Відносна %', fontsize=7)
        ax.set_title(f"{row['author']}\n"
                     f"Слів: {row['total_words']:.0f} | Токс: {row['toxicity_score']:.2f}",
                     fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Показуємо ОРИГІНАЛЬНІ значення
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    for idx in range(len(df_sorted), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Профілі емоцій всіх авторів',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out('all_emotion_profiles.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_toxicity_profiles(df):
    """Профілі токсичності"""
    if df.empty:
        return

    cats = ['Токсичність', 'Негатив', 'Гнів', 'Страх', 'Маніпуляції']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]

    df_sorted = df.sort_values('toxicity_score', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    rows, cols = (n + 3) // 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4), subplot_kw=dict(projection='polar'))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx >= len(axes):
            break

        vals = [row['toxicity_score'] * 100,
                row['негатив'],
                row['гнів'],
                row['страх'],
                row['manipulation']]

        max_val = max(vals) if max(vals) > 0 else 1
        norm_vals = [v / max_val * 100 for v in vals] + [vals[0] / max_val * 100]

        axes[idx].plot(angles, norm_vals, 'o-', color='darkred', linewidth=2)
        axes[idx].fill(angles, norm_vals, alpha=0.25, color='coral')
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(cats, fontsize=8)
        axes[idx].set_ylim(0, 100)
        axes[idx].set_title(f"{row['author']}\n"
                            f"Токс: {row['toxicity_score']:.2f}",
                            fontsize=9, fontweight='bold')
        axes[idx].grid(True)

    for idx in range(len(df_sorted), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Профілі токсичності',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out('all_toxicity_profiles.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_common_radar(df):
    """Спільна радарна діаграма"""
    cats = ['Токсичність', 'Негатив', 'Гнів', 'Страх', 'Маніпуляції']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

    # Знаходимо максимум для нормалізації
    max_all = max(df['toxicity_score'].max() * 100,
                  df['негатив'].max(),
                  df['гнів'].max(),
                  df['страх'].max(),
                  df['manipulation'].max())

    for idx, (_, row) in enumerate(df.sort_values('toxicity_score', ascending=False).iterrows()):
        vals = [row['toxicity_score'] * 100,
                row['негатив'],
                row['гнів'],
                row['страх'],
                row['manipulation']]

        norm_vals = [v / max_all * 100 for v in vals] + [vals[0] / max_all * 100]

        ax.plot(angles, norm_vals, 'o-', lw=1.5,
                label=f"{row['author']} (токс: {row['toxicity_score']:.2f})",
                color=colors[idx])
        ax.fill(angles, norm_vals, alpha=0.05, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title('Спільний профіль токсичності',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out('common_radar.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_emotions_comparison(df):
    """Порівняння всіх емоцій"""
    emotions = ['позитив', 'радість', 'довіра', 'очікування',
                'негатив', 'гнів', 'страх', 'сум', 'огида', 'здивування']
    labels = ['Позитив', 'Радість', 'Довіра', 'Очікування',
              'Негатив', 'Гнів', 'Страх', 'Сум', 'Огида', 'Здивування']

    plot_df = df[['author'] + emotions].copy()
    plot_df['toxicity'] = df['toxicity_score']
    plot_df = plot_df.sort_values('toxicity', ascending=False).drop('toxicity', axis=1)
    plot_df.columns = ['author'] + labels

    ax = plot_df.set_index('author').plot(kind='bar', figsize=(18, 8), width=0.8)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel('Частота', fontsize=12)
    plt.title('Порівняння всіх емоцій', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9, ncol=2)
    plt.grid(axis='y', alpha=0.3)

    ax.set_xticklabels([f"{a}\n(токс: {df[df['author'] == a]['toxicity_score'].values[0]:.2f})"
                        for a in plot_df['author'].values], rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(out('all_emotions_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_balance(df):
    """Емоційний баланс"""
    df_sorted = df.sort_values('toxicity_score', ascending=False)
    x = range(len(df_sorted))

    plt.figure(figsize=(16, 7))
    plt.bar([i - 0.2 for i in x], df_sorted['позитив'], 0.4,
            label='Позитив', color='green', edgecolor='black', alpha=0.8)
    plt.bar([i + 0.2 for i in x], df_sorted['негатив'], 0.4,
            label='Негатив', color='red', edgecolor='black', alpha=0.8)

    plt.xticks(x, [f"{a}\n(токс: {t:.2f})" for a, t in zip(df_sorted['author'], df_sorted['toxicity_score'])],
               rotation=45, ha='right', fontsize=9)
    plt.ylabel('Частота', fontsize=12)
    plt.title('Емоційний баланс: Позитив vs Негатив', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out('balance.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_top_categories(df, category_key, title, color, filename):
    """Топ-10 категорій"""
    counter = Counter()
    for _, row in df.iterrows():
        if isinstance(row[category_key], dict):
            counter.update(row[category_key])

    if counter:
        top = counter.most_common(10)
        plt.figure(figsize=(12, 6))

        labels = [t[0][:40] + '...' if len(t[0]) > 40 else t[0] for t in top]
        values = [t[1] for t in top]

        plt.barh(labels, values, color=color, alpha=0.7)
        plt.xlabel('Кількість', fontsize=12)
        plt.title(f'Топ-10 {title}', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        for i, (_, v) in enumerate(top):
            plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(out(filename), dpi=150, bbox_inches='tight')
        plt.show()


# ОСНОВНА ПРОГРАМА
def main():
    df = load_files()

    # ТОП-5 найтоксичніших
    df_sorted = df.sort_values('toxicity_score', ascending=False)
    print("ТОП-5 НАЙТОКСИЧНІШИХ:")
    for _, r in df_sorted.head(5).iterrows():
        print(f"{r['author']:15} | Токс: {r['toxicity_score']:.2f} | "
              f"Негатив: {r['негатив']:4.0f} | Маніп: {r['manipulation']:4.0f}")

    print("НАЙМЕНШ ТОКСИЧНІ:")
    for _, r in df_sorted.tail(5).iterrows():
        print(f"{r['author']:15} | Токс: {r['toxicity_score']:.2f}")

    # Генерація всіх графіків
    plot_bar(df, 'toxicity_score', 'Ступінь токсичності',
             'Токсичність', 'salmon', 'raw_toxicity.png', round_digits=4, ylim=(0, 1.1))
    plot_all_emotion_profiles(df)
    plot_all_toxicity_profiles(df)
    plot_common_radar(df)
    plot_all_emotions_comparison(df)
    plot_bar(df, 'emotional_intensity', 'Емоційна інтенсивність (сума всіх емоцій)',
             'Інтенсивність', 'teal', 'emotional_intensity.png')
    plot_balance(df)
    plot_bar(df, 'manipulation', 'Рівень маніпуляцій',
             'Маніпуляції', 'purple', 'manipulation.png')

    # Створення Excel-файлу
    with pd.ExcelWriter(out('toxicity_analysis.xlsx')) as writer:
        main_df = df[['author', 'total_words', 'sentences', 'vocab_size',
                      'toxicity_score', 'manipulation', 'emotional_intensity',
                      'позитив', 'радість', 'довіра', 'очікування',
                      'негатив', 'гнів', 'страх', 'сум', 'огида', 'здивування',
                      'positive_negative_ratio']].copy()
        main_df.to_excel(writer, sheet_name='Всі_дані', index=False)

    print(f"\nГотово. Результати збережено в папці {OUT_DIR}/")


if __name__ == "__main__":
    main()