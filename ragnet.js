import 'cheerio'
import 'dotenv/config'
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { AlibabaTongyiEmbeddings } from '@langchain/community/embeddings/alibaba_tongyi'
import { ChatAlibabaTongyi } from '@langchain/community/chat_models/alibaba_tongyi'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { Annotation, StateGraph } from '@langchain/langgraph'

// 网站URL配置 - 可以轻松更换不同的网站
const url = 'https://nba.hupu.com/stats/players'

// 使用CheerioWebBaseLoader加载网页内容
const cheerioLoader = new CheerioWebBaseLoader(url, {
  // 尝试获取更多内容，包括meta标签和title
  textContent: false,  // 保留HTML结构
  selector: 'body',    // 选择body标签
})

const docs = await cheerioLoader.load()

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,  // 减小chunk大小，提高检索精度
  chunkOverlap: 150,  // 增加重叠，确保信息不丢失
  separators: ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]  // 优化分割符
})
const allSplits = await splitter.splitDocuments(docs)
const embeddings = new AlibabaTongyiEmbeddings({})
const vectorStore = new MemoryVectorStore(embeddings)
await vectorStore.addDocuments(allSplits)

const llm = new ChatAlibabaTongyi({
  model: 'qwen-max',
  temperature: 0,
  apiKey: process.env.ALIBABA_API_KEY,
})

// 定义agent工作流
const StateAnnotation = Annotation.Root({
  question: Annotation, //用户输入的问题
  context: Annotation, //检索答案
  answer: Annotation, //回答最终结果
})

// 定义检索方法，通过用户输入的问题， 在vectorStore中检索相关文档，最终返回检索结果
const retrieve = async (state) => {
  let retrievedDocs = []

  // 策略1: 完整问题检索
  try {
    const docs1 = await vectorStore.similaritySearch(state.question, 5)
    retrievedDocs.push(...docs1)
  } catch (error) {
    // 检索失败时继续
    console.log('完整问题检索失败:', error.message)
  }

  // 策略2: 关键词检索
  const keywords = state.question.toLowerCase().split(/[\s,，。！？；、]+/).filter(word => word.length > 1)

  for (const keyword of keywords) {
    try {
      const docs2 = await vectorStore.similaritySearch(keyword, 3)
      docs2.forEach(doc => {
        if (!retrievedDocs.some(existing => existing.pageContent === doc.pageContent)) {
          retrievedDocs.push(doc)
        }
      })
    } catch (error) {
      // 检索失败时继续
      console.log('关键词检索失败:', error.message)
    }
  }

  return { context: retrievedDocs }
}

//定义生成答案的方法。先远程获取一个 promptTemplate （也可以手写），然后结合 retrievedDocs 一起生成 messages ，最后调用 llm 生成自然语言的结果
// 使用自定义的中文prompt模板
import { ChatPromptTemplate } from '@langchain/core/prompts'

const promptTemplate = ChatPromptTemplate.fromMessages([
  ['system', `你是一个专业的网站内容分析助手。请根据提供的网站文档内容，准确、全面地回答用户的问题。

核心原则：
1. 完全基于提供的文档内容回答问题，不得编造任何信息
2. 仔细分析文档中的所有相关信息，包括：
   - 明确提及的信息
   - 隐含的状态（如"空缺"、"暂无"、"正在招募"等）
   - HTML链接标签（<a href="...">链接文本</a>）和按钮文本的含义
   - 数字、日期、列表等具体信息
   - 状态描述和说明文字
   - 链接URL和链接文本的关系

3. 信息识别策略：
   - 如果文档提到某个概念但显示"空缺"、"暂无"、"正在招募"、"成为赞助商"等状态，要明确说明现状
   - 如果文档包含列表或具体信息，要准确列出
   - 如果文档提到链接但未提供详细信息，要说明需要访问链接获取更多信息
   - 如果文档中完全没有相关信息，要明确说明"在提供的文档中未找到相关信息"
   - 特别注意："成为赞助商"通常意味着该位置空缺，正在招募新的赞助商
   - 对于动态网站，如果只看到JavaScript代码，说明内容需要JavaScript渲染，要说明这个情况

4. 回答要求：
   - 优先提供具体、准确的信息
   - 如果信息不完整，要说明现状和限制
   - 保持客观、准确，避免推测
   - 用中文回答，语言简洁明了
   - 必要时引用文档中的具体表述
   - 如果有对话历史上下文，要结合之前的问题来理解当前问题的意图

5. 通用性要求：
   - 适用于任何网站内容，不依赖特定网站结构
   - 能够识别各种格式的信息（列表、表格、状态描述等）
   - 能够处理中英文混合内容
   - 能够理解上下文联系，回答相关问题`],
  ['user', `问题: {question}

网站文档内容:
{context}

请根据以上网站文档内容，准确回答用户的问题。如果文档中提到相关信息但状态不明确，请详细说明现状。`]
])

const generate = async (state) => {
  // 处理文档内容，提取链接信息
  let docsContent = state.context.map((doc) => {
    let content = doc.pageContent
    
    // 提取HTML链接信息
    const linkPattern = /<a[^>]*href=['"]([^'"]*)['"][^>]*>([^<]*)<\/a>/gi;
    const links = content.match(linkPattern);
    
    if (links) {
      // 在内容中添加链接说明
      content += '\n\n[发现的链接信息]:\n';
      links.forEach((link, index) => {
        content += `链接${index + 1}: ${link}\n`;
      });
    }
    
    return content;
  }).join('\n\n')
  
  // 添加上下文信息（对话历史）
  let contextInfo = ''
  if (conversationHistory.length > 1) {
    contextInfo = '\n\n[对话历史上下文]:\n'
    conversationHistory.slice(-3).forEach((item, index) => {
      contextInfo += `之前的问题${index + 1}: ${item.question}\n`
    })
  }
  
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent + contextInfo,
  })

  // 使用流式输出
  let fullAnswer = ''
  const stream = await llm.stream(messages)

  for await (const chunk of stream) {
    if (chunk.content) {
      // 逐字符输出，模拟打字机效果
      for (const char of chunk.content) {
        process.stdout.write(char)
        // 添加小延迟模拟真实流式效果
        await new Promise(resolve => setTimeout(resolve, 30))
      }
      fullAnswer += chunk.content
    }
  }

  return { answer: fullAnswer }
}

// 定义workflow工作流
const graph = new StateGraph(StateAnnotation)
  .addNode('retrieve', retrieve)
  .addNode('generate', generate)
  .addEdge('__start__', 'retrieve')
  .addEdge('retrieve', 'generate')
  .addEdge('generate', '__end__')
  .compile()

// 动态上下文联系 - 交互式问答
import readline from 'readline'

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})


// 存储对话历史，用于上下文联系
let conversationHistory = []

console.log('=== 动态上下文联系系统 ===')
console.log('输入 "exit" 或 "quit" 退出系统\n')

const askQuestion = () => {
  rl.question('请输入您的问题:  ', async (userInput) => {
    if (userInput.toLowerCase() === 'exit' || userInput.toLowerCase() === 'quit') {
      console.log('\n=== 感谢使用，再见！ ===')
      rl.close()
      return
    }
    
    if (userInput.trim() === '') {
      console.log('请输入有效的问题\n')
      askQuestion()
      return
    }
    
    console.log('\n--- 正在处理您的问题 ---')
    console.log('问题:', userInput)
    
    // 将当前问题添加到对话历史
    conversationHistory.push({
      question: userInput,
      timestamp: new Date().toISOString()
    })
    
    try {
      // 调用Agent处理问题
      let inputs = { question: userInput }
      await graph.invoke(inputs)
      
      console.log('\n' + '='.repeat(60) + '\n')
      
      // 继续询问下一个问题
      askQuestion()
    } catch (error) {
      console.log('处理问题时出错:', error.message)
      console.log('\n' + '='.repeat(60) + '\n')
      askQuestion()
    }
  })
}

// 开始交互
askQuestion()


// 自定义模板
// const generateAnswer = async(state) => {
//   console.log('生成答案:', state.context)
//   const promptTemplate = await ChatPromptTemplate.fromMessages([
//     ['system', '你是一个专业的知识库助手，请根据以下内容回答用户的问题。'],
//     ['user', '问题: {question}\n相关文档: {context}'],
//   ])
//   const messages = await promptTemplate.invoke({
//     question: state.question,
//     context: state.context,
//   })
// }