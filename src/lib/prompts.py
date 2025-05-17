from pathlib import Path
from typing import Dict, Any
from camel.prompts import TextPrompt
import jinja2

class JinjaPromptManager:
    """
    Класс для работы с промптами через Jinja templates.
    Загружает шаблоны промптов из указанной директории и предоставляет
    метод для рендеринга промптов по имени файла и параметрам.
    """
    
    def __init__(self, prompt_directory: str):
        """
        Инициализация менеджера промптов.
        
        Args:
            prompt_directory: Путь к директории с шаблонами промптов
        """
        self.prompts_dir = Path(prompt_directory)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.prompts_dir)),
            autoescape=jinja2.select_autoescape([]),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Загружает все шаблоны промптов из директории."""
        # Поддержка Jinja шаблонов
        for filename in self.prompts_dir.glob('*.j2'):
            key = filename.stem
            self.templates[key] = self.jinja_env.get_template(filename.name)
        
        for filename in self.prompts_dir.glob('*.jinja2'):
            key = filename.stem
            self.templates[key] = self.jinja_env.get_template(filename.name)
            
        # Поддержка текстовых файлов (для обратной совместимости)
        for filename in self.prompts_dir.glob('*.txt'):
            key = filename.stem
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            # Преобразуем обычный текст в Jinja шаблон
            self.templates[key] = jinja2.Template(content)
    
    def render_prompt(self, prompt_name: str, params: Dict[str, Any] = None) -> str:
        """
        Рендерит промпт по имени и параметрам.
        
        Args:
            prompt_name: Имя промпта (без расширения файла)
            params: Словарь с параметрами для рендеринга
            
        Returns:
            Отрендеренный текст промпта
        
        Raises:
            KeyError: Если промпт с указанным именем не найден
        """
        if params is None:
            params = {}
            
        if prompt_name not in self.templates:
            raise KeyError(f"Промпт с именем '{prompt_name}' не найден")
            
        return self.templates[prompt_name].render(**params)
    
    def get_text_prompt(self, prompt_name: str, params: Dict[str, Any] = None) -> TextPrompt:
        """
        Получает отрендеренный промпт в виде объекта TextPrompt из camel-ai.
        
        Args:
            prompt_name: Имя промпта (без расширения файла)
            params: Словарь с параметрами для рендеринга
            
        Returns:
            Объект TextPrompt с отрендеренным текстом
        """
        rendered_text = self.render_prompt(prompt_name, params)
        return TextPrompt(rendered_text)
