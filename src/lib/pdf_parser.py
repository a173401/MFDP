import pdfplumber
import layoutparser as lp
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator, Dict, Any, Union


@dataclass
class TextBlock:
    """Представляет текстовый блок, извлеченный из страницы PDF."""
    type: str  # Тип блока (Text, Title, List и т.д.)
    bbox: Tuple[float, float, float, float]  # Координаты ограничивающего прямоугольника (x0, y0, x1, y1)
    text: str  # Извлеченный текст


@dataclass
class PageData:
    """Представляет данные, извлеченные из одной страницы PDF."""
    page_num: int  # Номер страницы (с индексацией от 0)
    blocks: List[TextBlock]  # Структурированные текстовые блоки
    full_text: str  # Неструктурированный текст со всей страницы
    
    def to_markdown(self) -> str:
        """
        Преобразует данные страницы в формат Markdown для использования с LLM.
        
        Создает хорошо структурированное представление страницы PDF в формате Markdown,
        оптимизированное для анализа языковыми моделями. Сохраняет семантическую
        структуру документа и добавляет метаданные для лучшего понимания.
        
        Возвращает:
            str: Представление данных страницы в формате Markdown.
        """
        result = []
        
        # Заголовок страницы
        result.append(f"# Page {self.page_num + 1}")
        result.append("")
        
        # Добавляем метаданные о странице
        if self.blocks:
            block_types = {}
            for block in self.blocks:
                block_types[block.type] = block_types.get(block.type, 0) + 1
            
            metadata = []
            for block_type, count in block_types.items():
                metadata.append(f"{count} {block_type}")
            
            result.append(f"**Page Structure:** {', '.join(metadata)}")
            result.append("")
        
        # Добавляем структурированные блоки, если они есть
        if self.blocks:
            # Сортируем блоки по их вертикальной позиции (сверху вниз)
            sorted_blocks = sorted(self.blocks, key=lambda b: b.bbox[1])
            
            for i, block in enumerate(sorted_blocks):
                # Форматирование в зависимости от типа блока
                if block.type == "Title":
                    result.append(f"## {block.text.strip()}")
                elif block.type == "Text":
                    result.append(f"### Text Block {i+1}")
                    result.append("")
                    result.append(block.text.strip())
                elif block.type == "List":
                    result.append(f"### List {i+1}")
                    result.append("")
                    
                    # Пытаемся определить, нумерованный это список или маркированный
                    list_items = block.text.strip().split('\n')
                    if any(item.strip() and item.strip()[0].isdigit() and
                          ('.' in item.strip()[:3] or ')' in item.strip()[:3])
                          for item in list_items if item.strip()):
                        # Нумерованный список
                        for item in list_items:
                            if item.strip():
                                # Проверяем, имеет ли элемент уже формат нумерованного списка
                                if item.strip()[0].isdigit() and ('.' in item.strip()[:3] or ')' in item.strip()[:3]):
                                    result.append(item.strip())
                                else:
                                    result.append(f"1. {item.strip()}")
                    else:
                        # Маркированный список
                        for item in list_items:
                            if item.strip():
                                # Проверяем, имеет ли элемент уже формат маркированного списка
                                if item.strip().startswith('-') or item.strip().startswith('*') or item.strip().startswith('•'):
                                    result.append(item.strip())
                                else:
                                    result.append(f"- {item.strip()}")
                elif block.type == "Table":
                    result.append(f"### Table {i+1}")
                    result.append("")
                    result.append("```")
                    result.append(block.text.strip())
                    result.append("```")
                elif block.type == "Figure":
                    result.append(f"### Figure {i+1}")
                    result.append("")
                    if block.text.strip():
                        result.append(f"*Caption:* {block.text.strip()}")
                    else:
                        result.append("*[Figure without text description]*")
                else:
                    # Для других типов блоков
                    result.append(f"### {block.type} Block {i+1}")
                    result.append("")
                    result.append(block.text.strip())
                
                # Добавляем пустую строку после каждого блока
                result.append("")
        if self.full_text:
            result.append("## Unstructured Extracted Text")
            result.append("")
            result.append(self.full_text.strip())
            result.append("")
        
        # Объединяем все строки с правильными переносами
        return "\n".join(result)


class PDFParser:
    """
    Класс для парсинга PDF-документов с целью извлечения текста с информацией о макете
    и преобразования страниц в изображения.
    
    Этот парсер использует layoutparser для обнаружения элементов макета и pdfplumber
    для извлечения текста из PDF-документов.
    """
    
    def __init__(self, model_name: Optional[str] = None, model_config_file_name: Optional[str] = None):
        """
        Инициализация PDFParser с опциональной моделью макета.
        
        Аргументы:
            model_name (str, опционально): Название модели для определения макета.
                Если None, используется модель PubLayNet по умолчанию.
            model_config_file_name (str, опционально): Путь к файлу конфигурации модели.
                Если None, используется конфигурация по умолчанию.
        """
        # Конфигурация модели по умолчанию
        # У Detectron2 для данной модели есть дефект, скаченная модель хранится под неправильным именем. 
        # Для его устранения необходимо передать пути до файла модели и файла конфигурации самостоятельно.
        default_model = 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config'
        default_label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        
        # Инициализация модели макета
        if model_name:
            self.model = lp.Detectron2LayoutModel(
                model_config_file_name,
                model_name,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=default_label_map
            )
        else:
            self.model = lp.Detectron2LayoutModel(
                default_model,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                label_map=default_label_map
            )
    
    def parse_pages(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Generator[PageData, None, None]:
        """
        Парсинг PDF-документа по страницам, извлечение как структурированного,
        так и неструктурированного текста.
        
        Аргументы:
            pdf_path (str): Путь к PDF-файлу
            page_numbers (list, опционально): Список номеров страниц для обработки.
                Если None, обрабатываются все страницы.
                
        Возвращает (генератор):
            PageData: Экземпляр класса, содержащий структурированные и 
                неструктурированные данные для каждой страницы.
            
        Вызывает исключения:
            FileNotFoundError: Если PDF-файл не существует.
            ValueError: Если любой из указанных номеров страниц недействителен.
        """
        try:
            # Открыть PDF-файл
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Если page_numbers не указан, обрабатываем все страницы
                if page_numbers is None:
                    page_numbers = range(total_pages)
                else:
                    # Проверяем, что номера страниц действительны
                    invalid_pages = [p for p in page_numbers if p < 0 or p >= total_pages]
                    if invalid_pages:
                        raise ValueError(f"Недействительные номера страниц: {invalid_pages}. Допустимый диапазон: 0-{total_pages-1}")
                
                # Обработка каждой страницы
                for page_num in page_numbers:
                    try:
                        page = pdf.pages[page_num]
                        
                        # Извлечение неструктурированного текста со всей страницы
                        full_text = page.extract_text() or ""
                        
                        # Преобразование PDF-страницы в изображение для layoutparser
                        img = page.to_image(resolution=300)
                        image = np.array(img.original)
                        
                        # Определение макета с помощью layoutparser
                        layout = self.model.detect(image)
                        
                        # Получение всех текстовых блоков без фильтрации по типу
                        text_blocks = layout
                        
                        page_width, page_height = page.width, page.height
                        image_height, image_width = image.shape[:2]
                        
                        # Коэффициенты масштабирования для преобразования координат из изображения в PDF
                        x_scale = page_width / image_width
                        y_scale = page_height / image_height
                        
                        blocks = []
                        
                        for block in text_blocks:
                            # Преобразование координат layoutparser в координаты pdfplumber
                            x0 = block.block.x_1 * x_scale
                            y0 = block.block.y_1 * y_scale
                            x1 = block.block.x_2 * x_scale
                            y1 = block.block.y_2 * y_scale
                            
                            # Создание ограничивающего прямоугольника для pdfplumber
                            bbox = (x0, y0, x1, y1)
                            
                            # Извлечение текста из ограничивающего прямоугольника
                            crop = page.crop(bbox)
                            text = crop.extract_text() or ""
                            
                            # Добавляем только если есть текст
                            if text:
                                blocks.append(TextBlock(
                                    type=block.type,
                                    bbox=bbox,
                                    text=text
                                ))
                        
                        # Возвращаем данные страницы
                        yield PageData(
                            page_num=page_num,
                            blocks=blocks,
                            full_text=full_text
                        )
                    except Exception as e:
                        # Логируем ошибку и продолжаем со следующей страницей
                        print(f"Ошибка при обработке страницы {page_num}: {str(e)}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF-файл не найден: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка при парсинге PDF: {str(e)}")
    
    def parse_pages_as_images(self, pdf_path: str, page_numbers: Optional[List[int]] = None, 
                             resolution: int = 300) -> Generator[Image.Image, None, None]:
        """
        Преобразование страниц PDF в объекты PIL Image.
        
        Аргументы:
            pdf_path (str): Путь к PDF-файлу
            page_numbers (list, опционально): Список номеров страниц для обработки.
                Если None, обрабатываются все страницы.
            resolution (int, опционально): Разрешение выходных изображений в DPI.
                
        Возвращает (генератор):
            Image: Объект PIL Image для каждой страницы.
            
        Вызывает исключения:
            FileNotFoundError: Если PDF-файл не существует.
            ValueError: Если любой из указанных номеров страниц недействителен.
        """
        try:
            # Открыть PDF-файл
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Если page_numbers не указан, обрабатываем все страницы
                if page_numbers is None:
                    page_numbers = range(total_pages)
                else:
                    # Проверяем, что номера страниц действительны
                    invalid_pages = [p for p in page_numbers if p < 0 or p >= total_pages]
                    if invalid_pages:
                        raise ValueError(f"Недействительные номера страниц: {invalid_pages}. Допустимый диапазон: 0-{total_pages-1}")
                
                # Обработка каждой страницы
                for page_num in page_numbers:
                    try:
                        page = pdf.pages[page_num]
                        
                        # Преобразование PDF-страницы в изображение
                        img = page.to_image(resolution=resolution)
                        
                        # Возвращаем объект PIL Image
                        yield img.original
                    except Exception as e:
                        # Логируем ошибку и продолжаем со следующей страницей
                        print(f"Ошибка при преобразовании страницы {page_num} в изображение: {str(e)}")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF-файл не найден: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка при парсинге PDF: {str(e)}")


# Пример использования
if __name__ == "__main__":
    # Создаем парсер с моделью по умолчанию
    parser = PDFParser()
    
    # Пример: Обработка первых 5 страниц тестового PDF
    pdf_path = "datasets/raw/brandbooks/brand1_Anoma-BrandGuide.pdf"
    
    print("Извлечение текста из страниц PDF...")
    for i, page_data in enumerate(parser.parse_pages(pdf_path, page_numbers=list(range(5)))):
        print(f"Страница {page_data.page_num + 1}")
        print(f"Длина полного текста: {len(page_data.full_text)}")
        print(f"Структурированных блоков: {len(page_data.blocks)}")
        
        for j, block in enumerate(page_data.blocks[:3]):  # Показываем только первые 3 блока
            print(f"  Блок {j+1}: Тип: {block.type}, Текст: {block.text[:50]}...")
        
        print("-" * 50)
    
    print("\nПреобразование страниц PDF в изображения...")
    for i, img in enumerate(parser.parse_pages_as_images(pdf_path, page_numbers=[0])):
        print(f"Размер изображения страницы {i+1}: {img.size}")
        # При необходимости можно сохранить изображение
        # img.save(f"page_{i+1}.png")