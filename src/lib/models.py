import numpy as np
from pydantic import BaseModel
from typing import Optional


class BrandbookRules(BaseModel):
    rules: list[str]
    
    def rules_to_markdown(self) -> str:
        """
        Преобразует только правила брендбука в markdown формат для передачи в LLM.
        
        Returns:
            str: Строка в markdown формате, содержащая только правила брендбука.
        """
        md = "# Правила брендбука\n\n"
        
        if not self.rules:
            md += "_Правила отсутствуют_\n"
            return md
            
        for rule in self.rules:
            md += f"- {rule}\n"
            
        return md
    

class ClassifiedRules(BaseModel):
    logo_rules: list[str] = []
    color_system_rules: list[str] = []
    typography_rules: list[str] = []
    restrictions_rules: list[str] = []
    composition_rules: list[str] = []
    legal_rules: list[str] = []
    other_rules: list[str] = []
    
    def _rules_to_markdown(self, rules: list[str], header: str) -> str:
        """Вспомогательный метод для преобразования списка правил в формат markdown."""
        if not rules:
            return ""
        md = f"### {header}\n"
        for rule in rules:
            md += f"- {rule}\n"
        return md
    
    def logo_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.logo_rules, "Logo Rules")
    
    def color_system_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.color_system_rules, "Color System Rules")
    
    def typography_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.typography_rules, "Typography Rules")
    
    def restrictions_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.restrictions_rules, "Restrictions Rules")
    
    def composition_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.composition_rules, "Composition Rules")
    
    def legal_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.legal_rules, "Legal Rules")
    
    def other_rules_to_markdown(self) -> str:
        return self._rules_to_markdown(self.other_rules, "Other Rules")
    
    def to_markdown(self) -> str:
        """Преобразует все правила в markdown представление."""
        sections = [
            self.logo_rules_to_markdown(),
            self.color_system_rules_to_markdown(),
            self.typography_rules_to_markdown(),
            self.restrictions_rules_to_markdown(),
            self.composition_rules_to_markdown(),
            self.legal_rules_to_markdown(),
            self.other_rules_to_markdown()
        ]
        # Отфильтровываем пустые секции и объединяем с двойным переводом строки
        return "\n\n".join([section for section in sections if section])


class Color(BaseModel):
    r: int
    g: int
    b: int
    pantone: str

    def to_numpy(self) -> np.array:
        return np.array([self.r, self.g, self.b], dtype=np.uint32)

class LLMColorFontResponse(BaseModel):
    colors_rgb: list[Color]
    fonts: list[str]

class BrandbookInfo(BaseModel):
    classified_rules: ClassifiedRules
    colors_rgb: list[Color]
    fonts: list[str] 

class LLMResponsePrompt(BaseModel):
    prompt: str

