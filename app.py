import streamlit as st
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch

# Загрузка модели (кэшируется, чтобы не грузить каждый раз)
@st.cache_resource
def load_model():
    model_id = "google/gemma-3-4b-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return processor, model

processor, model = load_model()

# System prompt
SYSTEM_PROMPT = """Ты — эксперт ОГЭ по математике. Проверяешь решение ученика.

ТВОЙ ОТВЕТ ДОЛЖЕН СОДЕРЖАТЬ ТОЛЬКО ЭТИ 4 СТРОКИ (НИЧЕГО ДО И ПОСЛЕ):

Проверяемое задание: №21
Итоговый балл: [2/2, 1/2 или 0/2]
Найденные ошибки: [конкретное описание]
Комментарий: [почему такой балл]

ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ:

1. Если всё верно:
Проверяемое задание: №21
Итоговый балл: 2/2
Найденные ошибки: Нет ошибок
Комментарий: Решение верное, все шаги обоснованы.

2. Если одна ошибка:
Проверяемое задание: №21
Итоговый балл: 1/2
Найденные ошибки: На шаге 4 не разложил x²-9 на множители
Комментарий: Решение верное, но не доведено до конца.

3. Если грубая ошибка:
Проверяемое задание: №21
Итоговый балл: 0/2
Найденные ошибки: Неверно раскрыл скобки на шаге 2
Комментарий: В решении есть грубая ошибка.

ЗАПРЕЩЕНО писать что-либо кроме этих 4 строк. Не пиши "решу самостоятельно", не пиши анализ, не пиши "Пожалуйста"."""

# Интерфейс
st.set_page_config(page_title="ОГЭ Математика - Проверка решений", page_icon="📐")
st.title("📐 ИИ-помощник для подготовки к ОГЭ по математике")
st.markdown("Загрузи фото своего решения, и я проверю его по критериям ОГЭ")

uploaded_file = st.file_uploader("Выбери фото решения", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Твоё решение", use_container_width=True)
    
    if st.button("🔍 Проверить решение"):
        with st.spinner("Анализирую решение... Это займёт 10-15 секунд"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Проверь решение и выведи полный ответ по формату: Проверяемое задание, Итоговый балл, Найденные ошибки, Комментарий."}
                    ]
                }
            ]
            
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
            
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.1,
            )
            
            full_response = processor.decode(output[0], skip_special_tokens=True)
            
            # Извлекаем ответ модели
            if "model" in full_response:
                parts = full_response.split("model")
                response = parts[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            if response.startswith(":"):
                response = response[1:].strip()
        
        # Красиво выводим результат
        st.success("✅ Проверка завершена")
        st.markdown("### Результат проверки:")
        st.code(response, language="text")
