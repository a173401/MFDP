from pathlib import Path
import pandas as pd
import re
import json
from typing import Dict, List, Optional

from .models import BrandbookInfo, ClassifiedRules, Color


class BrandDataset:
    """
    Класс для работы с датасетом брендбуков и соответствующих им изображений.
    
    Класс парсит дерево директорий и предоставляет API для работы с данными датасета.
    Соответствие между брендбуками и изображениями устанавливается по префиксу имени файла
    (например, brand1_ как у брендбука, так и у изображений).
    
    Примеры использования:
    ```python
    from pathlib import Path
    
    # Инициализация датасета
    dataset = BrandDataset(Path("."))
    
    # Получение списка путей к брендбукам
    brandbooks = dataset.get_brandbooks()
    
    # Получение пар "брендбук - изображения"
    pairs = dataset.get_brand_image_pairs()
    
    # Получение DataFrame с данными
    df = dataset.get_dataframe()
    ```
    """
    
    def __init__(
        self,
        root_dir: Path,
        brandbooks_subdir: str = "raw/brandbooks",
        images_subdir: str = "raw/images/compliant",
        processed_subdir: str = "processed"
    ) -> None:
        """
        Инициализирует объект BrandDataset.
        
        Args:
            root_dir: Корневая директория проекта
            brandbooks_subdir: Поддиректория с брендбуками относительно корневой
            images_subdir: Поддиректория с изображениями относительно корневой
            processed_subdir: Поддиректория с обработанными данными относительно корневой
            
        Raises:
            FileNotFoundError: Если указанные директории не существуют
            TypeError: Если root_dir не является объектом Path
        """
        if not isinstance(root_dir, Path):
            raise TypeError("root_dir должен быть объектом типа Path")
            
        self.root_dir = root_dir
        self.brandbooks_dir = root_dir / brandbooks_subdir
        self.images_dir = root_dir / images_subdir
        self.processed_dir = root_dir / processed_subdir
        
        # Проверка наличия директорий
        if not self.brandbooks_dir.exists():
            raise FileNotFoundError(f"Директория с брендбуками не найдена: {self.brandbooks_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена: {self.images_dir}")
        
        # Проверка наличия директории processed - не обязательна
        # Если директория не существует, создаем ее
        if not self.processed_dir.exists():
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэширование списков файлов и префиксов
        self._brandbooks: Optional[List[Path]] = None
        self._images: Optional[List[Path]] = None
        self._brand_prefixes: Optional[Dict[Path, str]] = None
        self._brandbook_info_cache: Dict[str, Optional[BrandbookInfo]] = {}
    
    def get_brandbooks(self) -> List[Path]:
        """
        Возвращает список путей к брендбукам.
        
        Returns:
            Список путей к файлам брендбуков
        """
        if self._brandbooks is None:
            self._brandbooks = list(self.brandbooks_dir.glob("*.pdf"))
        
        return self._brandbooks
    
    def get_images(self) -> List[Path]:
        """
        Возвращает список путей к изображениям.
        
        Returns:
            Список путей к файлам изображений
        """
        if self._images is None:
            self._images = list(self.images_dir.glob("*.png"))
        
        return self._images
    
    def _get_brand_prefixes(self) -> Dict[Path, str]:
        """
        Получает словарь соответствия путей брендбуков и их префиксов.
        
        Returns:
            Словарь {путь_к_брендбуку: префикс}
        """
        if self._brand_prefixes is None:
            self._brand_prefixes = {}
            for brandbook in self.get_brandbooks():
                prefix = self._extract_prefix(brandbook.name)
                if prefix:
                    self._brand_prefixes[brandbook] = prefix
        
        return self._brand_prefixes
    
    def _extract_prefix(self, filename: str) -> Optional[str]:
        """
        Извлекает префикс бренда из имени файла.
        
        Args:
            filename: Имя файла
        
        Returns:
            Префикс бренда или None, если префикс не найден
        """
        # Пример: из "brand1_Anoma-BrandGuide.pdf" извлечь "brand1_"
        match = re.match(r'(brand\d+_)', filename)
        if match:
            return match.group(1)
        return None
    
    def get_brand_image_pairs(self) -> Dict[Path, List[Path]]:
        """
        Возвращает пары "брендбук-изображения".
        
        Returns:
            Словарь, где ключ - путь к брендбуку,
            значение - список путей к соответствующим изображениям
        """
        result = {}
        brandbooks_prefixes = self._get_brand_prefixes()
        images = self.get_images()
        
        for brandbook, prefix in brandbooks_prefixes.items():
            # Находим все изображения, имена которых начинаются с этим префиксом
            matching_images = [img for img in images if img.name.startswith(prefix)]
            result[brandbook] = matching_images
        
        return result
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с данными о брендбуках и изображениях.
        
        Returns:
            DataFrame с колонками:
                - brand_id: идентификатор бренда (например, "brand1")
                - brandbook_path: путь к файлу брендбука
                - image_path: путь к файлу изображения
                
        Примечание: Для каждого файла с изображением создается отдельная строка в DataFrame.
        """
        rows = []
        brandbooks_prefixes = self._get_brand_prefixes()
        
        for brandbook, prefix in brandbooks_prefixes.items():
            brand_id = prefix.rstrip('_')  # Удаляем "_" из конца префикса
            
            # Находим все изображения с этим префиксом
            matching_images = [img for img in self.get_images() if img.name.startswith(prefix)]
            
            for image in matching_images:
                rows.append({
                    'brand_id': brand_id,
                    'brandbook_path': brandbook,
                    'image_path': image
                })
        
        return pd.DataFrame(rows)
    
    def get_brandbook(self, brandbook_prefix: str) -> Path:
        """
        Возвращает путь к файлу брендбука по префиксу.

        Args:
            brandbook_prefix: Префикс имени файла брендбука

        Returns:
            Путь к файлу брендбука

        Raises:
            FileNotFoundError: Если брендбук с указанным префиксом не найден
        """
        brandbook = [item for item in self.get_brandbooks() if item.name.startswith(brandbook_prefix)]
        if not brandbook:
            raise FileNotFoundError(f"Брендбук с префиксом '{brandbook_prefix}' не найден")
        return brandbook[0]
        
    def get_brandbook_info_path(self, brand_prefix: str) -> Optional[Path]:
        """
        Возвращает путь к JSON-файлу с информацией о брендбуке.
        
        Args:
            brand_prefix: Префикс бренда (например, "brand1_")
            
        Returns:
            Path: Путь к JSON-файлу или None, если файл не существует
        """
        # Удаляем "_" из конца префикса, если он есть
        brand_id = brand_prefix.rstrip('_')
        
        # Проверяем, существует ли файл с правилами
        json_path = self.processed_dir / f"{brand_id}_extracted_rules.json"
        
        if json_path.exists():
            return json_path
        return None
        
    def get_brandbook_info(self, brand_prefix: str) -> Optional[BrandbookInfo]:
        """
        Возвращает объект BrandbookInfo для указанного бренда.
        
        Args:
            brand_prefix: Префикс бренда (например, "brand1_" или "brand1")
            
        Returns:
            BrandbookInfo: Объект с информацией о брендбуке или None, если информация отсутствует
        """
        # Используем кэш, если информация уже была загружена
        if brand_prefix in self._brandbook_info_cache:
            return self._brandbook_info_cache[brand_prefix]
            
        # Получаем путь к JSON-файлу
        json_path = self.get_brandbook_info_path(brand_prefix)
        
        if json_path is None:
            # Если JSON-файл не существует, кэшируем None и возвращаем None
            self._brandbook_info_cache[brand_prefix] = None
            return None
            
        try:
            # Читаем JSON-файл и десериализуем в объект BrandbookInfo
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                info = BrandbookInfo(**data)
                
            # Кэшируем результат
            self._brandbook_info_cache[brand_prefix] = info
            return info
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # В случае ошибки при чтении или парсинге, возвращаем None
            self._brandbook_info_cache[brand_prefix] = None
            return None
    
    def get_dataframe_with_info(self) -> pd.DataFrame:
        """
        Возвращает расширенный DataFrame с данными о брендбуках, изображениях и информацией из JSON.
        
        Returns:
            DataFrame с колонками:
                - brand_id: идентификатор бренда (например, "brand1")
                - brandbook_path: путь к файлу брендбука
                - image_path: путь к файлу изображения
                - classified_rules: объект ClassifiedRules или None
                - colors_rgb: список объектов Color или None
                - fonts: список шрифтов или None
                - logo_rules: список правил для логотипа или None
                - color_system_rules: список правил для цветовой системы или None
                - typography_rules: список правил для типографики или None
                - restrictions_rules: список правил для ограничений или None
                - composition_rules: список правил для композиции или None
                - legal_rules: список правил для юридических аспектов или None
                - other_rules: список прочих правил или None
                
        Примечание: Для каждого файла с изображением создается отдельная строка в DataFrame.
        """
        rows = []
        brandbooks_prefixes = self._get_brand_prefixes()
        
        for brandbook, prefix in brandbooks_prefixes.items():
            brand_id = prefix.rstrip('_')  # Удаляем "_" из конца префикса
            
            # Получаем информацию о брендбуке
            brandbook_info = self.get_brandbook_info(brand_id)
            
            # Находим все изображения с этим префиксом
            matching_images = [img for img in self.get_images() if img.name.startswith(prefix)]
            
            for image in matching_images:
                row = {
                    'brand_id': brand_id,
                    'brandbook_path': brandbook,
                    'image_path': image,
                }
                
                # Добавляем информацию из объекта BrandbookInfo, если она доступна
                if brandbook_info:
                    # Добавляем основные поля
                    row['classified_rules'] = brandbook_info.classified_rules
                    row['colors_rgb'] = brandbook_info.colors_rgb
                    row['fonts'] = brandbook_info.fonts
                    
                    # Добавляем поля из ClassifiedRules
                    row['logo_rules'] = brandbook_info.classified_rules.logo_rules
                    row['color_system_rules'] = brandbook_info.classified_rules.color_system_rules
                    row['typography_rules'] = brandbook_info.classified_rules.typography_rules
                    row['restrictions_rules'] = brandbook_info.classified_rules.restrictions_rules
                    row['composition_rules'] = brandbook_info.classified_rules.composition_rules
                    row['legal_rules'] = brandbook_info.classified_rules.legal_rules
                    row['other_rules'] = brandbook_info.classified_rules.other_rules
                else:
                    # Если информация недоступна, заполняем поля значениями None
                    row['classified_rules'] = None
                    row['colors_rgb'] = None
                    row['fonts'] = None
                    row['logo_rules'] = None
                    row['color_system_rules'] = None
                    row['typography_rules'] = None
                    row['restrictions_rules'] = None
                    row['composition_rules'] = None
                    row['legal_rules'] = None
                    row['other_rules'] = None
                
                rows.append(row)
        
        return pd.DataFrame(rows)
