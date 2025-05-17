import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
from collections import Counter
from src.lib.pdf_parser import PDFParser
import colorsys
import logging
import time
import cv2
import colour
from functools import wraps

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('color_analyzer')

# Декоратор для измерения времени выполнения функций
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Начало выполнения {func.__name__}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Завершение {func.__name__}, время выполнения: {execution_time:.2f} сек.")
        return result
    return wrapper

@log_execution_time
def extract_colors(image: Image.Image, n_colors: int = 10,
                   min_saturation: float = 0.1,
                   ignore_whites: bool = True,
                   ignore_blacks: bool = True) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Извлекает основные цвета из изображения.
    
    Args:
        image: PIL изображение
        n_colors: максимальное количество цветов для извлечения
        min_saturation: минимальная насыщенность цвета (для фильтрации почти серых)
        ignore_whites: игнорировать ли белые цвета
        ignore_blacks: игнорировать ли черные цвета
        
    Returns:
        Список кортежей (цвет, процент), где цвет - RGB-кортеж (r,g,b),
        а процент - доля этого цвета в изображении
    """
    # Преобразуем изображение в numpy массив
    logger.debug(f"Преобразование изображения размером {image.width}x{image.height} в numpy массив")
    img_array = np.array(image)
    
    # Если изображение имеет альфа-канал, убираем его
    if img_array.shape[2] == 4:
        logger.debug("Обнаружен альфа-канал, удаляем его")
        img_array = img_array[:, :, :3]
    
    # Уменьшаем размер изображения для ускорения обработки, если оно большое
    # if image.width > 1000 or image.height > 1000:
    #     new_width = min(image.width, 1000)
    #     new_height = min(image.height, 1000)
    #     logger.info(f"Изображение слишком большое, уменьшаем до {new_width}x{new_height}")
    #     resized_img = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    #     img_array = np.array(resized_img)
    
    # Преобразуем 3D массив (высота, ширина, RGB) в 2D массив (пиксели, RGB)
    logger.debug("Преобразование 3D массива в 2D массив пикселей")
    pixels = img_array.reshape(-1, 3)
    total_pixels = len(pixels)
    logger.debug(f"Всего пикселей для анализа: {total_pixels}")
    
    # Создаем список для хранения цветов и их долей
    dominant_colors = []
    
    # Создаем маску для фильтрации слишком светлых/темных цветов
    mask = np.ones(len(pixels), dtype=bool)
    
    if ignore_whites:
        # Фильтруем белые цвета (все RGB значения > 240)
        logger.debug("Фильтрация белых цветов")
        white_mask = ~np.all(pixels > 240, axis=1)
        mask = mask & white_mask
        
    if ignore_blacks:
        # Фильтруем черные цвета (все RGB значения < 15)
        logger.debug("Фильтрация черных цветов")
        black_mask = ~np.all(pixels < 15, axis=1)
        mask = mask & black_mask
    
    # Фильтруем пиксели по насыщенности
    if min_saturation > 0:
        # Преобразуем RGB в HSV для каждого пикселя
        logger.debug(f"Фильтрация пикселей по насыщенности (мин. {min_saturation})")
        logger.debug("Преобразование RGB в HSV")

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0
        hsv[..., 1:] /= 255.0
        hsv_pixels = hsv.reshape(-1, 3)
        saturation_mask = hsv_pixels[:, 1] >= min_saturation
        mask = mask & saturation_mask
    
    filtered_pixels = pixels[mask]
    logger.info(f"После фильтрации осталось {len(filtered_pixels)} пикселей " +
                f"({len(filtered_pixels)/total_pixels*100:.1f}% от исходного количества)")
    
    # Если после фильтрации осталось слишком мало пикселей
    if len(filtered_pixels) < 100:
        logger.warning("После фильтрации осталось слишком мало пикселей, используем все пиксели")
        filtered_pixels = pixels  # Используем все пиксели
    
    # Квантизация цветов: объединяем похожие цвета
    # Для этого округляем значения RGB до ближайших 10
    # logger.debug("Выполнение квантизации цветов")
    # quantized = (filtered_pixels // 10) * 10
    
    # Подсчитываем частоту каждого цвета
    logger.debug("Подсчет частоты каждого цвета")

    concatenated = (filtered_pixels[:, 0].astype(np.uint32)) + (filtered_pixels[:, 1].astype(np.uint32) << 8) + (filtered_pixels[:, 2].astype(np.uint32) << 16)
    values, counts = np.unique(concatenated, return_counts=True)
    index_of_sorted = counts.argsort()[::-1]  # Индексы, сортированные по убыванию количества)
    top_values = values[index_of_sorted]
    top_rgb_values = np.array([(value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF) for value in top_values])

    logger.debug(f"Найдено {len(top_rgb_values)} уникальных цветов после квантизации")
    total_filtered_pixels = len(filtered_pixels)
    for index, rgb_value in enumerate(top_rgb_values[:n_colors]):
        logger.debug(f"Цвет {index + 1}: RGB({rgb_value[0]}, {rgb_value[1]}, {rgb_value[2]})")
        count = counts[index_of_sorted][index]
        percentage = count / total_filtered_pixels
        dominant_colors.append((rgb_value, percentage))
        logger.debug(f"Добавлен цвет RGB{rgb_value} с долей {percentage:.3f}")

    
    return dominant_colors

@log_execution_time
def find_common_colors(all_pages_colors: List[List[Tuple[Tuple[int, int, int], float]]],
                       min_presence: float = 0.5,
                       similarity_threshold: float = 30) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Находит общие цвета, которые встречаются на нескольких страницах.
    
    Args:
        all_pages_colors: список списков цветов для каждой страницы
        min_presence: минимальная доля страниц, на которых должен присутствовать цвет
        similarity_threshold: порог сходства для объединения похожих цветов (по Евклидову расстоянию)
        
    Returns:
        Список кортежей (цвет, вес), где вес - комбинированный показатель частоты и распространённости
    """
    if not all_pages_colors:
        return []
    
    # Получаем все уникальные цвета со всех страниц
    all_colors = []
    logger.info(f"Анализ цветов из {len(all_pages_colors)} страниц")
    for i, page_colors in enumerate(all_pages_colors):
        logger.debug(f"Страница {i+1}: {len(page_colors)} цветов")
        all_colors.extend([color for color, _ in page_colors])
    
    logger.info(f"Всего собрано {len(all_colors)} цветов со всех страниц")
    
    # Группируем похожие цвета
    grouped_colors = []
    grouped_weights = []
    
    logger.info(f"Группировка похожих цветов с порогом сходства {similarity_threshold}")
    for i, color in enumerate(all_colors):
        if i % 100 == 0 and i > 0:
            logger.debug(f"Обработано {i}/{len(all_colors)} цветов ({i/len(all_colors)*100:.1f}%)")
        
        # Проверяем, близок ли цвет к уже найденным группам
        found_group = False
        for i, group_color in enumerate(grouped_colors):
            # Евклидово расстояние между цветами
            distance = np.sqrt(np.sum((np.array(color) - np.array(group_color))**2))
            if distance < similarity_threshold:
                # Обновляем центр группы (среднее между текущим центром и новым цветом)
                grouped_colors[i] = tuple((np.array(group_color) + np.array(color)) // 2)
                grouped_weights[i] += 1
                found_group = True
                break
        
        if not found_group:
            # Создаем новую группу
            grouped_colors.append(color)
            grouped_weights.append(1)
    
    logger.info(f"Создано {len(grouped_colors)} групп цветов")
    
    # Считаем, на скольких страницах встречается каждый цвет
    pages_count = len(all_pages_colors)
    color_presence = {}
    
    logger.info("Анализ присутствия цветов на страницах")
    for group_idx, group_color in enumerate(grouped_colors):
        if group_idx % 10 == 0 and group_idx > 0:
            logger.debug(f"Обработано {group_idx}/{len(grouped_colors)} групп цветов")
            
        presence_count = 0
        
        for page_idx, page_colors in enumerate(all_pages_colors):
            # Проверяем, есть ли на странице цвет, похожий на текущую группу
            for page_color, _ in page_colors:
                distance = np.sqrt(sum((np.array(page_color) - np.array(group_color))**2))
                if distance < similarity_threshold:
                    presence_count += 1
                    break
        
        # Считаем долю страниц, где встречается цвет
        presence_ratio = presence_count / pages_count
        color_presence[group_idx] = presence_ratio
        logger.debug(f"Цвет RGB{group_color} присутствует на {presence_count}/{pages_count} страницах ({presence_ratio:.2f})")
    
    # Фильтруем цвета по минимальному порогу присутствия
    common_colors = []
    logger.info(f"Фильтрация цветов по минимальному порогу присутствия ({min_presence})")
    for group_idx, group_color in enumerate(grouped_colors):
        if color_presence[group_idx] >= min_presence:
            # Вес = (частота * доля страниц)
            weight = grouped_weights[group_idx] * color_presence[group_idx]
            common_colors.append((group_color, weight))
            logger.debug(f"Цвет RGB{group_color} прошел фильтрацию с весом {weight:.3f}")
    
    # Сортируем по весу по убыванию
    common_colors.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Найдено {len(common_colors)} общих цветов после фильтрации")
    
    return common_colors

@log_execution_time
def analyze_color_zones(pages_images: List[Image.Image],
                        grid_size: Tuple[int, int] = (3, 3)) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Анализирует распределение цветов по зонам страницы.
    
    Args:
        pages_images: список PIL изображений страниц
        grid_size: размер сетки для разделения страницы на зоны (строки, столбцы)
        
    Returns:
        Словарь, где ключи - названия зон (например, 'top', 'bottom', 'left', 'right', 'center'),
        а значения - списки доминантных цветов в этих зонах
    """
    rows, cols = grid_size
    zones = {
        'top': [],
        'bottom': [],
        'left': [],
        'right': [],
        'center': [],
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }
    
    logger.info(f"Анализ цветовых зон для {len(pages_images)} изображений с сеткой {grid_size}")
    
    for img_idx, img in enumerate(pages_images):
        logger.debug(f"Обработка изображения {img_idx+1}/{len(pages_images)}")
        width, height = img.size
        
        # Разделяем изображение на зоны
        row_height = height // rows
        col_width = width // cols
        logger.debug(f"Размеры ячейки сетки: {col_width}x{row_height} пикселей")
        
        # Извлекаем цвета из каждой зоны
        for r in range(rows):
            for c in range(cols):
                x0 = c * col_width
                y0 = r * row_height
                x1 = x0 + col_width
                y1 = y0 + row_height
                
                # Вырезаем зону
                zone_img = img.crop((x0, y0, x1, y1))
                logger.debug(f"Анализ зоны ({r},{c}): координаты ({x0},{y0},{x1},{y1})")
                
                # Определяем, к какой логической зоне относится эта часть сетки
                zone_name = None
                
                # Центральная ячейка
                if r == rows // 2 and c == cols // 2:
                    zone_name = 'center'
                # Верхний ряд
                elif r == 0:
                    if c == 0:
                        zone_name = 'top_left'
                    elif c == cols - 1:
                        zone_name = 'top_right'
                    else:
                        zone_name = 'top'
                # Нижний ряд
                elif r == rows - 1:
                    if c == 0:
                        zone_name = 'bottom_left'
                    elif c == cols - 1:
                        zone_name = 'bottom_right'
                    else:
                        zone_name = 'bottom'
                # Левый столбец
                elif c == 0:
                    zone_name = 'left'
                # Правый столбец
                elif c == cols - 1:
                    zone_name = 'right'
                else:
                    # Остальные ячейки считаем центральными
                    zone_name = 'center'
                
                logger.debug(f"Зона ({r},{c}) отнесена к логической зоне '{zone_name}'")
                
                # Извлекаем цвета из зоны
                logger.debug(f"Извлечение цветов из зоны '{zone_name}'")
                zone_colors = extract_colors(zone_img, n_colors=3)
                
                # Добавляем только сами цвета (без процентов) в соответствующую зону
                for color, percentage in zone_colors:
                    zones[zone_name].append(color)
                    logger.debug(f"Добавлен цвет RGB{color} с процентом {percentage:.3f} в зону '{zone_name}'")
    
    # Для каждой зоны находим наиболее часто встречающиеся цвета
    logger.info("Определение наиболее частых цветов для каждой зоны")
    for zone_name, colors in zones.items():
        if colors:
            logger.debug(f"Зона '{zone_name}': найдено {len(colors)} цветов")
            
            # Квантизируем цвета для объединения похожих
            quantized = [(color[0]//10*10, color[1]//10*10, color[2]//10*10) for color in colors]
            
            # Считаем частоту каждого цвета
            color_counter = Counter(quantized)
            
            # Получаем 5 самых частых цветов
            most_common = color_counter.most_common(5)
            zones[zone_name] = [color for color, count in most_common]
            logger.debug(f"Зона '{zone_name}': топ-5 цветов: {zones[zone_name]}")
        else:
            logger.debug(f"Зона '{zone_name}': цвета не найдены")
            zones[zone_name] = []
    
    return zones

@log_execution_time
def analyze_color_patterns(pages_images: List[Image.Image]) -> Tuple[
    List[Tuple[Tuple[int, int, int], float]],
    Dict[str, List[Tuple[int, int, int]]]
]:
    """
    Анализирует цветовые паттерны по всем страницам брендбука.
    
    Args:
        pages_images: список PIL изображений страниц
        
    Returns:
        Кортеж из двух элементов:
        1. Список общих цветов с их весами
        2. Словарь с цветами по зонам страницы
    """
    # Анализ цветов по всем страницам
    logger.info(f"Начало анализа цветовых паттернов для {len(pages_images)} страниц")
    all_pages_colors = []
    for i, page_img in enumerate(pages_images):
        logger.info(f"Извлечение цветов из страницы {i+1}/{len(pages_images)}")
        page_colors = extract_colors(page_img)
        all_pages_colors.append(page_colors)
        logger.debug(f"Страница {i+1}: найдено {len(page_colors)} основных цветов")
    
    # Выявление повторяющихся цветов
    logger.info("Поиск общих цветов на всех страницах")
    common_colors = find_common_colors(all_pages_colors)
    
    # Анализ распределения цветов внутри страниц
    logger.info("Анализ распределения цветов по зонам страниц")
    # color_zones = analyze_color_zones(pages_images)
    color_zones = {}
    return common_colors, color_zones

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Преобразует цвет из RGB в HEX."""
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def rgb_to_string(rgb: Tuple[int, int, int]) -> str:
    """Преобразует цвет из RGB в строку формата 'r,g,b'."""
    return f'{rgb[0]}, {rgb[1]}, {rgb[2]}'

def color_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Преобразует RGB цвет в цветовое пространство LAB.
    
    Args:
        rgb: Кортеж RGB значений (r, g, b) в диапазоне 0-255
        
    Returns:
        Кортеж LAB значений (L, a, b)
    """
    # Нормализация RGB значений до 0-1
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # Преобразование в XYZ
    r = ((r > 0.04045) and ((r + 0.055) / 1.055) ** 2.4) or (r / 12.92)
    g = ((g > 0.04045) and ((g + 0.055) / 1.055) ** 2.4) or (g / 12.92)
    b = ((b > 0.04045) and ((b + 0.055) / 1.055) ** 2.4) or (b / 12.92)
    
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    # Нормализация XYZ по белой точке D65
    x /= 0.95047
    y /= 1.0
    z /= 1.08883
    
    # Преобразование XYZ в LAB
    x = (x > 0.008856) and x ** (1/3) or (7.787 * x + 16/116)
    y = (y > 0.008856) and y ** (1/3) or (7.787 * y + 16/116)
    z = (z > 0.008856) and z ** (1/3) or (7.787 * z + 16/116)
    
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    return (L, a, b)

def color_distance(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """
    Вычисляет расстояние между двумя цветами в пространстве LAB (Delta E).
    
    Args:
        rgb1, rgb2: Кортежи RGB значений (r, g, b) в диапазоне 0-255
        
    Returns:
        Расстояние Delta E между цветами
    """
    # Преобразуем RGB в LAB
    lab1 = color_to_lab(rgb1)
    lab2 = color_to_lab(rgb2)
    
    # Вычисляем Delta E (CIE76)
    delta_e = np.sqrt((lab1[0] - lab2[0])**2 + (lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2)
    
    return delta_e

def color_distance_v2(rgb1: np.array, rgb2: np.array) -> float:
    """
    Вычисляет цветовое расстояние между двумя RGB цветами используя пространство CIE Lab.
    
    Параметры:
        rgb1 (np.Array): Первый цвет в формате RGB (значения от 0 до 255)
        rgb2 (np.Array): Второй цвет в формате RGB (значения от 0 до 255)
        
    Возвращает:
        float: Значение Delta E - мера визуального различия между двумя цветами в пространстве Lab
    """
    xyz_1 = colour.sRGB_to_XYZ(rgb1 / 255.)
    xyz_2 = colour.sRGB_to_XYZ(rgb2 / 255.)
    lab_1 = colour.XYZ_to_Lab(xyz_1)
    lab_2 = colour.XYZ_to_Lab(xyz_2)
    return colour.delta_E(lab_1, lab_2)



def visualize_color_palette(colors: List[Tuple[Tuple[int, int, int], float]], 
                           output_path: str = None,
                           show_percentages: bool = True,
                           width: int = 600,
                           bar_height: int = 50) -> Image.Image:
    """
    Создает визуализацию цветовой палитры.
    
    Args:
        colors: Список кортежей (цвет, вес)
        output_path: Путь для сохранения изображения (опционально)
        show_percentages: Показывать ли проценты
        width: Ширина изображения
        bar_height: Высота цветовой полосы
        
    Returns:
        PIL.Image: Изображение с визуализацией палитры
    """
    # Если список цветов пуст, возвращаем пустое изображение
    if not colors:
        img = Image.new('RGB', (width, bar_height), color=(255, 255, 255))
        if output_path:
            img.save(output_path)
        return img
    
    # Определяем высоту изображения
    height = len(colors) * bar_height
    
    # Создаем новое изображение
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Получаем шрифт для текста
    try:
        font = ImageFont.truetype("arial.ttf", size=bar_height // 3)
    except IOError:
        font = ImageFont.load_default()
    
    # Нормализуем веса, чтобы они давали в сумме 1.0
    total_weight = sum(weight for _, weight in colors)
    normalized_colors = [(color, weight / total_weight) for color, weight in colors]
    
    # Рисуем цветовые полосы
    y_offset = 0
    for i, (color, weight) in enumerate(normalized_colors):
        # Рисуем полосу цвета
        draw.rectangle([0, y_offset, width, y_offset + bar_height], fill=color)
        
        # Добавляем текст с информацией о цвете
        # hex_color = rgb_to_hex(color)
        text_of_color = rgb_to_string(color)
        # Определяем, должен ли текст быть темным или светлым
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # Формируем текст
        if show_percentages:
            text = f"{text_of_color} ({int(weight * 100)}%)"
        else:
            text = text_of_color
        
        # Размещаем текст по центру полосы
        # В новых версиях PIL используем textbbox вместо устаревшего textsize
        try:
            # Для новых версий PIL
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Запасной вариант для старых версий PIL
            text_width, text_height = getattr(draw, 'textsize', lambda t, font: (10, 10))(text, font=font)
            
        text_position = ((width - text_width) // 2, y_offset + (bar_height - text_height) // 2)
        
        draw.text(text_position, text, fill=text_color, font=font)
        
        y_offset += bar_height
    
    # Сохраняем изображение, если указан путь
    if output_path:
        img.save(output_path)
    
    return img

# Импорт для функции визуализации
try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    # Если не удалось импортировать, создаем заглушки
    class ImageDraw:
        @staticmethod
        def Draw(image):
            class DummyDraw:
                def rectangle(self, *args, **kwargs): pass
                def text(self, *args, **kwargs): pass
                def textbbox(self, *args, **kwargs): return (0, 0, 10, 10)
                def textsize(self, *args, **kwargs): return (10, 10)  # Для обратной совместимости
            return DummyDraw()
    
    class ImageFont:
        @staticmethod
        def truetype(*args, **kwargs): return None
        
        @staticmethod
        def load_default(): return None

# Функция для применения анализа к брендбуку
@log_execution_time
def analyze_brandbook_colors(pdf_path: str,
                              parser: PDFParser,
                              page_numbers: Optional[List[int]] = None,
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Полный анализ цветов брендбука.
    
    Args:
        pdf_path: Путь к PDF-файлу брендбука
        page_numbers: Номера страниц для анализа (если None, анализируются все)
        output_dir: Директория для сохранения визуализаций (если None, не сохраняются)
        
    Returns:
        Словарь с результатами анализа:
        - common_colors: общие цвета во всем брендбуке
        - color_zones: распределение цветов по зонам страницы
        - visualization_path: путь к сохраненной визуализации (если output_dir указан)
    """
    import os
    
    logger.info(f"Начало анализа брендбука: {pdf_path}")
    if page_numbers:
        logger.info(f"Анализ страниц: {page_numbers}")
    else:
        logger.info("Анализ всех страниц брендбука")
    
    # Получаем изображения страниц
    logger.info("Получение изображений страниц из PDF")
    start_time = time.time()
    pages_images = list(parser.parse_pages_as_images(pdf_path, page_numbers))
    end_time = time.time()
    logger.info(f"Получено {len(pages_images)} изображений страниц за {end_time-start_time:.2f} сек.")
    
    # Анализируем цветовые паттерны
    logger.info("Начало анализа цветовых паттернов")
    common_colors, color_zones = analyze_color_patterns(pages_images)
    
    # Создаем результирующий словарь
    result = {
        'common_colors': common_colors,
        'color_zones': color_zones,
        'visualization_path': None
    }
    
    # Если указана директория для вывода, сохраняем визуализации
    if output_dir and common_colors:
        logger.info(f"Сохранение визуализаций в директорию: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Имя файла брендбука без пути и расширения
        brandbook_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Путь для сохранения визуализации
        viz_path = os.path.join(output_dir, f"{brandbook_name}_palette.png")
        logger.info(f"Создание визуализации палитры: {viz_path}")
        
        # Создаем и сохраняем визуализацию
        visualize_color_palette(common_colors, viz_path)
        
        # Добавляем путь к визуализации в результат
        result['visualization_path'] = viz_path
        logger.info(f"Визуализация сохранена: {viz_path}")
    
    return result

if __name__ == "__main__":
    # Настройка дополнительного логирования для консоли при запуске как main
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Пример использования
    logger.info("Запуск примера анализа брендбука")
    parser = PDFParser()
    brandbook_path = "datasets/raw/brandbooks/brand1_Anoma-BrandGuide.pdf"
    
    logger.info(f"Анализ брендбука: {brandbook_path}")
    results = analyze_brandbook_colors(brandbook_path, parser=parser, output_dir="./output")
    
    print(f"Найдено {len(results['common_colors'])} общих цветов")
    for i, (color, weight) in enumerate(results['common_colors'][:5]):
        print(f"Цвет {i+1}: RGB{color} - HEX: {rgb_to_hex(color)} (вес: {weight:.3f})")
    
    print("\nРаспределение цветов по зонам:")
    for zone, colors in results['color_zones'].items():
        if colors:
            print(f"Зона '{zone}': {len(colors)} цветов")
            for color in colors[:2]:  # Показываем только первые 2 цвета
                print(f"  - RGB{color} - HEX: {rgb_to_hex(color)}")
    
    logger.info("Анализ брендбука завершен")