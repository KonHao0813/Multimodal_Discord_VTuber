import discord
import asyncio
from discord.ext import commands
import os
import edge_tts
import datetime
import logging
from dotenv import load_dotenv
from ai_core import OptimizedMultimodalAIVtuber, PersonalityManager

# 載入環境變數
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
TEXT_CHANNEL_ID = int(os.getenv('TEXT_CHANNEL_ID'))

class MultimodalDiscordVTuber(commands.Bot):
    def __init__(self):
        print("正在初始化 MultimodalDiscordVTuber...")
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.ai = OptimizedMultimodalAIVtuber()  # 這裡不需要修改，因為預設值就是 "Qwen/Qwen-1_8B-Chat-Int4"
        print("self.ai 已初始化")
        self._my_voice_clients = {}
        self.text_channel_id = TEXT_CHANNEL_ID # 將全域變數的值賦給實例屬性
        self.setup_commands()

    def setup_commands(self):
        @self.command()
        async def join(ctx):
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                voice_client = await channel.connect()
                self._my_voice_clients[ctx.guild.id] = voice_client  # 使用新的屬性名稱
                await ctx.send(f"🎤 已加入 {channel.name}")
            else:
                await ctx.send("⚠️ 請先加入語音頻道！")

        @self.command()
        async def leave(ctx):
            if ctx.guild.id in self._my_voice_clients:  # 使用新的屬性名稱
                await self._my_voice_clients[ctx.guild.id].disconnect()  # 使用新的屬性名稱
                del self._my_voice_clients[ctx.guild.id]  # 使用新的屬性名稱
                await ctx.send("🚪 已離開語音頻道")
            else:
                await ctx.send("⚠️ 我不在語音頻道中")

        @self.command()
        async def mode(ctx, mode_name: str):
            personalities = PersonalityManager.get_all_personalities()
            if mode_name in personalities:
                self.ai.personality = mode_name
                await ctx.send(f"🔄 已切換至 {mode_name} 模式 - {personalities[mode_name]}")
            else:
                personality_list = "\n".join([f"• `{k}`: {v}" for k, v in personalities.items()])
                await ctx.send(f"❌ 無效模式，可用選項：\n{personality_list}")

        @self.command()
        async def forget(ctx):
            self.ai.memory.history.clear()
            self.ai.memory._save_memory()
            await ctx.send("🧹 記憶已清除")

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.channel.id == self.text_channel_id:
            try:
                if message.content.startswith("!"):
                    await self.process_commands(message)
                else:
                    image_url = None
                    if message.attachments:
                        for attachment in message.attachments:
                            if attachment.content_type.startswith('image/'):
                                image_url = attachment.url
                                break

                    print(f"在調用 process_multimodal_message 之前，message.content: {message.content}, image_url: {image_url}") # 添加這行
                    await self.process_multimodal_message(message, image_url)
                    print(f"在調用 process_commands 之前") # 添加這行
                    print(f"在調用 process_commands 之後") # 添加這行

            except Exception as e:
                logging.error(f"處理訊息時發生錯誤: {str(e)}")
                print(f"錯誤類型: {type(e)}") # 添加這行
                await message.channel.send("抱歉，處理訊息時出現了錯誤。")

    async def process_multimodal_message(self, message, image_url=None):
        print(f"進入 process_multimodal_message，message.content: {message.content}, image_url: {image_url}") # 添加這行
        if image_url:
            await message.channel.send("🔍 正在分析圖片...")
        print(f"調用 generate_response 之前") # 添加這行
        response = self.ai.generate_response(message.content, image_url)
        print(f"調用 generate_response 之後，response 類型: {type(response)}, response: {response}") # 添加這行
        channel = self.get_channel(self.text_channel_id) # 使用實例屬性
        if image_url:
            await channel.send(f"🎙️ **{message.author.display_name}**：{message.content} [附帶圖片]\n🤖 **虛擬主播**：{response}")
        else:
            await channel.send(f"🎙️ **{message.author.display_name}**：{message.content}\n🤖 **虛擬主播**：{response}")
        if message.guild.id in self._my_voice_clients:  # 使用新的屬性名稱
            print(f"調用 process_voice 之前，response 類型: {type(response)}, response: {response}") # 添加這行
            await self.process_voice(response, message.guild.id)
            print(f"調用 process_voice 之後") # 添加這行
        print(f"離開 process_multimodal_message") # 添加這行

    async def process_voice(self, text, guild_id):
        voice_client = self._my_voice_clients[guild_id]  # 使用新的屬性名稱
        audio_file = await asyncio.to_thread(lambda: asyncio.run(self.generate_audio(text)))
        audio_source = discord.FFmpegPCMAudio(audio_file)
        voice_client.play(audio_source, after=lambda e: self.cleanup_audio(audio_file))

    async def generate_audio(self, text):
        import time # 匯入 time 模組
        output_file = f"temp_{time.time()}.ogg" # 使用 time.time() 產生時間戳
        communicate = edge_tts.Communicate(text, voice='zh-TW-HsiaoChenNeural', rate="+15%", pitch="+30Hz")
        await communicate.save(output_file)
        return output_file

    def cleanup_audio(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    async def on_ready(self):
        print(f'✅ 已登入為：{self.user}')
        print(f'🌟 記憶系統狀態：')
        if hasattr(self, 'ai'):
            print(f'- 加載記憶數：{len(self.ai.memory.history)}')
        else:
            print("錯誤：self.ai 未被定義")

if __name__ == "__main__":
    bot = MultimodalDiscordVTuber()
    bot.run(TOKEN)