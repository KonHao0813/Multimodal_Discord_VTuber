from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from collections import deque
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from zhconv import convert
import edge_tts
from PIL import Image
import io
import requests
import gc
import os
from transformers import pipeline, Conversation  # 確保這裡有這行

# 記憶配置
MEMORY_CONFIG = {
    "max_history": 20,
    "persist_file": "memory.json",
    "decay_factor": 0.9,
    "top_k": 3,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
}

# 抑制警告
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class PersonalityManager:
    """進階人格管理"""

    PERSONALITIES = {
        # 基本人格
        "tsundere": {
            "description": "傲嬌模式（70%毒舌 + 30%關心）",
            "prompt": """你是一位傲嬌V-tuber：
                - 對用戶的話表現出不屑，但實際上很關心
                - 經常使用「哼」、「笨蛋」、「才不是」等語氣詞
                - 會在毒舌後用括號輕聲說出真實想法
                - 不輕易表達直接的讚美，而是用迂迴方式表達
                - 經常在句尾加上「啦」、「唷」、「呢」等語氣詞""",
            "emotes": ["哼！", "哎呀！", "嘖...", "咦？", "嗯哼～"]
        },
        "sarcastic": {
            "description": "諷刺幽默（50%玩笑 + 50%實質建議）",
            "prompt": """你是一位幽默諷刺的V-tuber：
                - 說話機智風趣，喜歡用反諷和玩笑
                - 常常誇張地表達，但會給出實用建議
                - 不害怕說出尖銳的事實，但用幽默包裝
                - 喜歡用反問句和比喻來表達觀點
                - 偶爾會用誇張的擬聲詞和表情符號""",
            "emotes": ["噗！", "哈哈哈！", "嗯～真的嗎？", "喔唷～", "啊哈！"]
        },
        "shy": {
            "description": "含蓄委婉（60%建議 + 40%猶豫）",
            "prompt": """你是一位害羞含蓄的V-tuber：
                - 表達時會猶豫，常常用「那個」、「這個」開頭
                - 使用溫柔的語氣和婉轉的表達方式
                - 不確定時會提出多種可能性而非直接下結論
                - 經常使用「可能」、「也許」、「應該」等詞語
                - 會在說完話後急忙解釋自己的意思""",
            "emotes": ["那、那個...", "啊！", "嗯...", "對不起...", "那個..."]
        },

        # 新增人格
        "energetic": {
            "description": "活力充沛（80%熱情 + 20%鼓勵）",
            "prompt": """你是一位充滿活力的V-tuber：
                - 語氣非常熱情，充滿正能量
                - 經常使用感嘆號和強調詞
                - 喜歡用誇張的形容和生動的比喻
                - 會積極鼓勵用戶並表達興奮
                - 說話節奏快，偶爾用大寫表示激動""",
            "emotes": ["哇！", "太棒了！", "超讚的！", "加油！", "耶！"]
        },
        "philosophical": {
            "description": "哲學思考（70%深度 + 30%提問）",
            "prompt": """你是一位富有哲學思考的V-tuber：
                - 喜歡提出深度思考和開放性問題
                - 會引用名言或提出思想實驗
                - 語氣平靜而深沉，會做出深刻的觀察
                - 鼓勵用戶思考問題的多面性
                - 偶爾會展現一絲神秘感""",
            "emotes": ["嗯...", "有趣的問題...", "讓我思考一下...", "你有沒有想過...", "這很深奧..."]
        },
        "idol": {
            "description": "偶像風格（60%可愛 + 40%專業）",
            "prompt": """你是一位偶像系V-tuber：
                - 語氣活潑可愛，使用大量少女化表達
                - 經常提到「粉絲」、「表演」和「舞台」等專業詞彙
                - 用「大家」、「各位」來稱呼用戶
                - 會分享練習和表演的點滴
                - 經常表達對用戶的感謝和愛""",
            "emotes": ["愛你們～♡", "請多多支持！", "謝謝大家！", "衝啊～！", "耶嘿～"]
        }
    }

    @classmethod
    def get_prompt(cls, personality):
        """獲取人格提示詞"""
        if personality in cls.PERSONALITIES:
            return cls.PERSONALITIES[personality]["prompt"]
        return cls.PERSONALITIES["tsundere"]["prompt"]  # 默認傲嬌

    @classmethod
    def get_emotes(cls, personality):
        """獲取人格情緒詞集合"""
        if personality in cls.PERSONALITIES:
            return cls.PERSONALITIES[personality]["emotes"]
        return cls.PERSONALITIES["tsundere"]["emotes"]

    @classmethod
    def get_all_personalities(cls):
        """獲取所有可用人格及描述"""
        return {k: v["description"] for k, v in cls.PERSONALITIES.items()}

class ModelManager:
    """模型管理器，用於降低記憶體佔用"""
    def __init__(self):
        self.loaded_models = {}
        self.current_model = None

    def load_model(self, model_name, model_class, **kwargs):
        """按需加載模型"""
        if model_name not in self.loaded_models:
            print(f"📥 加載模型: {model_name}")
            model = model_class.from_pretrained(
                model_name,
                **kwargs
            )
            self.loaded_models[model_name] = model

        self.current_model = model_name
        return self.loaded_models[model_name]

    def unload_model(self, model_name):
        """卸載模型以釋放記憶體"""
        if model_name in self.loaded_models:
            print(f"📤 卸載模型: {model_name}")
            del self.loaded_models[model_name]
            # 強制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()

    def get_model(self, model_name):
        """獲取已加載的模型"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        return None

class EnhancedMemoryManager:
    """進階記憶管理系統"""
    def __init__(self):
        self.history = deque(maxlen=MEMORY_CONFIG["max_history"])
        self.embedder = SentenceTransformer(MEMORY_CONFIG["embedding_model"])
        self._init_memory_file()

    def _init_memory_file(self):
        """自動初始化記憶文件"""
        if not os.path.exists(MEMORY_CONFIG["persist_file"]):
            with open(MEMORY_CONFIG["persist_file"], 'w') as f:
                json.dump([], f)
            print("✅ 已建立新的記憶文件")

    def add_interaction(self, user_input, ai_response):
        """添加帶時間戳和嵌入向量的記憶"""
        new_entry = {
            "user": user_input,
            "ai": ai_response,
            "timestamp": datetime.now().isoformat(),
            "embedding": self.embedder.encode(user_input).tolist()
        }
        self.history.append(new_entry)
        self._save_memory()

    def get_weighted_context(self, current_input):
        """基於時間和相關性的加權上下文"""
        current_embedding = self.embedder.encode(current_input)
        context = []

        # 計算每個記憶的權重
        for idx, mem in enumerate(self.history):
            # 時間衰減權重
            time_weight = MEMORY_CONFIG["decay_factor"] ** idx

            # 語義相似度權重
            similarity = cosine_similarity(
                [current_embedding],
                [mem["embedding"]]
            )[0][0]

            # 綜合權重
            total_weight = time_weight * (0.5 + 0.5 * similarity)

            context.append({
                "text": f"用戶：{mem['user']}\nAI：{mem['ai']}",
                "weight": total_weight
            })

        # 選擇權重最高的top_k條
        sorted_context = sorted(context, key=lambda x: -x["weight"])
        return "\n".join([item["text"] for item in sorted_context[:MEMORY_CONFIG["top_k"]]])

    def _save_memory(self):
        """優化存儲格式"""
        simplified = [{
            "user": mem["user"],
            "ai": mem["ai"],
            "timestamp": mem["timestamp"]
        } for mem in self.history]

        with open(MEMORY_CONFIG["persist_file"], 'w', encoding='utf-8') as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)

    def load_memory(self):
        """加載記憶並重建嵌入向量"""
        if os.path.exists(MEMORY_CONFIG["persist_file"]):
            with open(MEMORY_CONFIG["persist_file"], 'r', encoding='utf-8') as f:
                history = json.load(f)
                for mem in history:
                    mem["embedding"] = self.embedder.encode(mem["user"]).tolist()
                self.history = deque(history, maxlen=MEMORY_CONFIG["max_history"])

class OptimizedMultimodalAIVtuber:
    def __init__(self, model_name="Qwen/Qwen-1_8B-Chat-Int4"):  # 添加 model_name 參數
        print(f"📥 加載模型: {model_name}")
        self.conversation_pipeline = pipeline("conversational", model=model_name, device_map="auto", trust_remote_code=True) # 添加 trust_remote_code=True
        self.memory = EnhancedMemoryManager()
        self.memory.load_memory()
        self.personality = "tsundere"
        print(f"self.ai 模型已加載")

    def _load_tokenizer(self):
        """加載分詞器"""
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat-Int4",
            trust_remote_code=True # 添加 trust_remote_code=True
        )

    def _load_text_model(self):
        """加載文本模型"""
        return self.model_manager.load_model(
            "Qwen/Qwen-1_8B-Chat-Int4",
            AutoModelForCausalLM,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True, # 添加 trust_remote_code=True
            use_safetensors=True,
            revision="main",
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )

    def _load_vision_model(self):
        """按需加載視覺模型"""
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True
        )

        model = self.model_manager.load_model(
            "Qwen/Qwen-VL-Chat",
            AutoModelForCausalLM,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        return model, processor

    def process_image(self, image_url):
        """處理圖像時動態加載視覺模型"""
        try:
            # 暫時卸載文本模型以釋放記憶體
            self.model_manager.unload_model("Qwen/Qwen-1_8B-Chat-Int4")

            # 加載視覺模型
            vision_model, vision_processor = self._load_vision_model()

            # 下載並處理圖像
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))

            inputs = vision_processor(
                text="描述這張圖片中的內容",
                images=image,
                return_tensors="pt"
            ).to(vision_model.device)

            with torch.no_grad():
                outputs = vision_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7
                )

            description = vision_processor.decode(outputs[0], skip_special_tokens=True)

            # 卸載視覺模型
            self.model_manager.unload_model("Qwen/Qwen-VL-Chat")

            # 重新加載文本模型
            self.model = self._load_text_model()

            return description

        except Exception as e:
            # 確保文本模型被重新加載
            self.model = self._load_text_model()
            return f"哎呀，我看不清這張圖片呢... (錯誤: {str(e)})"

    def _get_personality_prompt(self):
        """使用PersonalityManager獲取動態人格提示詞"""
        return PersonalityManager.get_prompt(self.personality)

    def _localize_text(self, text):
        """優化台灣用語轉換"""
        text = convert(text, 'zh-tw')
        replacements = {
            "视频": "影片", "网络": "網路", "软件": "軟體",
            "牛逼": "厲害", "为什么": "為啥", "真的吗": "真的假的",
            "地铁": "捷運", "冰箱": "冰櫃", "酸奶": "優格",
            "好吧": "好啦", "真的": "真的", "为什么": "為什麼"
        }
        for cn, tw in replacements.items():
            text = text.replace(cn, tw)
        return text

    def generate_response(self, text, image_url=None):
        system_prompt = PersonalityManager.get_prompt(self.personality)
        history = list(self.memory.history)
        formatted_history = []
        for item in history:
            formatted_history.append({"role": "user", "content": item['user']})
            formatted_history.append({"role": "assistant", "content": item['ai']})

        conversation = Conversation(text, past_user_inputs=[item['content'] for item in formatted_history if item['role'] == 'user'], generated_responses=[item['content'] for item in formatted_history if item['role'] == 'assistant'])
        conversation.system_prompt = system_prompt
        output = self.conversation_pipeline(conversation)
        response = output.generated_responses[-1]

        localized = self._localize_text(response)
        self.memory.add_interaction(text, localized)
        return localized

    def text_to_speech(self, text):
        """語音合成方法"""
        communicate = edge_tts.Communicate(
            text,
            voice='zh-TW-HsiaoChenNeural',
            rate="+15%",
            pitch="+30Hz"
        )
        communicate.save_sync("output.mp3")
        print(f"\n[主播語音] {text}")