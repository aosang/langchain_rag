import path from 'path'
import 'dotenv/config'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { AlibabaTongyiEmbeddings } from '@langchain/community/embeddings/alibaba_tongyi'
import { CloudClient } from "chromadb"
import { ChatAlibabaTongyi } from '@langchain/community/chat_models/alibaba_tongyi'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables'

const pdfPath = path.resolve('file/front.pdf')
const loader = new PDFLoader(pdfPath)

const docs = await loader.load()

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,  // 增加块大小以减少块数量
  chunkOverlap: 300,  // 适当增加重叠以保持上下文
})

// 初始化ChromaDB云端客户端
console.log('正在初始化ChromaDB云端客户端...')

// 设置环境变量作为备用方案
process.env.CHROMA_API_KEY = 'ck-8iVSQzSGn91SmXQ71b8DTYKqW6JHPkXWJAWv1MJTeyap'

// 尝试多种配置方式
let chromaClient
try {
  // 方式1：使用 apiKey 参数
  console.log('尝试使用 apiKey 参数...')
  chromaClient = new CloudClient({
    apiKey: 'ck-8iVSQzSGn91SmXQ71b8DTYKqW6JHPkXWJAWv1MJTeyap',
    tenant: '4612e9e0-dca8-49ce-9568-bf6a737ce063',
    database: 'miles'
  });
} catch (error1) {
  console.log('apiKey 方式失败，尝试 apikey 参数...')
  try {
    // 方式2：使用 apikey 参数（小写）
    chromaClient = new CloudClient({
      apikey: 'ck-8iVSQzSGn91SmXQ71b8DTYKqW6JHPkXWJAWv1MJTeyap',
      tenant: '4612e9e0-dca8-49ce-9568-bf6a737ce063',
      database: 'miles'
    });
  } catch (error2) {
    console.log('apikey 方式也失败，尝试 key 参数...')
    try {
      // 方式3：使用 key 参数
      chromaClient = new CloudClient({
        key: 'ck-8iVSQzSGn91SmXQ71b8DTYKqW6JHPkXWJAWv1MJTeyap',
        tenant: '4612e9e0-dca8-49ce-9568-bf6a737ce063',
        database: 'miles'
      });
    } catch (error3) {
      console.error('所有方式都失败了')
      console.error('错误1 (apiKey):', error1.message)
      console.error('错误2 (apikey):', error2.message)
      console.error('错误3 (key):', error3.message)
      throw error3
    }
  }
}

// 测试ChromaDB连接
try {
  console.log('测试ChromaDB连接...')
  const heartbeat = await chromaClient.heartbeat()
  console.log('ChromaDB连接成功:', heartbeat)
} catch (error) {
  console.error('ChromaDB连接失败:', error.message)
  console.error('详细错误:', error)
  throw error
}

const allSplits = await textSplitter.splitDocuments(docs)
console.log(`文档已分割为 ${allSplits.length} 个块`)

// 配额限制检查
const QUOTA_LIMIT = 600  // 更新为新的配额限制
if (allSplits.length > QUOTA_LIMIT) {
  console.log(`⚠️  警告: 文档块数量 (${allSplits.length}) 超过配额限制 (${QUOTA_LIMIT})`)
  console.log(`📦 将只处理前 ${QUOTA_LIMIT} 个文档块`)
  allSplits.splice(QUOTA_LIMIT) // 只保留前600个块
  console.log(`✂️  已截取至 ${allSplits.length} 个块`)
}

// 初始化嵌入模型
console.log('正在初始化嵌入模型...')
const embeddings = new AlibabaTongyiEmbeddings({})

// 集合名称
const collectionName = 'pdf_documents'

// 创建或获取ChromaDB集合
console.log('正在创建或获取ChromaDB集合...')
let collection
try {
  // 尝试获取已存在的集合
  collection = await chromaClient.getCollection({ name: collectionName })
  console.log(`找到已存在的集合: ${collectionName}`)
  
  // 检查现有记录数量
  try {
    const existingCount = await collection.count()
    console.log(`📊 集合中现有记录数量: ${existingCount}`)
    
    const remainingQuota = QUOTA_LIMIT - existingCount
    console.log(`📈 剩余配额: ${remainingQuota}`)
    
    if (remainingQuota <= 0) {
      console.log(`❌ 配额已满，无法添加新记录`)
      console.log(`💡 建议: 删除部分现有记录或升级配额`)
      process.exit(0)  // 使用 process.exit() 替代 return
    }
    
    if (allSplits.length > remainingQuota) {
      console.log(`⚠️  调整: 文档块数量 (${allSplits.length}) 超过剩余配额 (${remainingQuota})`)
      allSplits.splice(remainingQuota)
      console.log(`✂️  已调整至 ${allSplits.length} 个块`)
    }
  } catch (countError) {
    console.log('无法获取集合记录数量，继续执行...')
  }
  
} catch (error) {
  // 如果集合不存在，则创建新的集合
  console.log(`集合不存在，正在创建新集合: ${collectionName}`)
  collection = await chromaClient.createCollection({ 
    name: collectionName,
    metadata: { description: 'PDF文档向量存储集合' }
  })
  console.log(`成功创建集合: ${collectionName}`)
}

// 准备文档数据
console.log('正在准备文档数据...')
const documents = []
const metadatas = []
const ids = []

for (let i = 0; i < allSplits.length; i++) {
  const doc = allSplits[i]
  documents.push(doc.pageContent)
  metadatas.push({
    source: doc.metadata.source || 'unknown',
    page: doc.metadata.page || 0,
    chunk_index: i
  })
  ids.push(`doc_${i}`)
}

// 生成嵌入向量
console.log('正在生成嵌入向量...')
const embedVectors = await embeddings.embedDocuments(documents)
console.log(`生成了 ${embedVectors.length} 个嵌入向量`)

// 将文档添加到ChromaDB
try {
  console.log('正在将文档添加到ChromaDB...')
  await collection.add({
    ids: ids,
    embeddings: embedVectors,
    metadatas: metadatas,
    documents: documents
  })
  console.log(`成功添加 ${documents.length} 个文档块到ChromaDB`)
} catch (error) {
  console.error('添加文档到ChromaDB失败:', error.message)
  console.error('详细错误:', error)
  throw error
}

// 执行相似性搜索
try {
  console.log('执行相似性搜索...')
  const query = 'When was Nike incorporated?'
  
  // 生成查询向量
  const queryVector = await embeddings.embedQuery(query)
  
  // 在ChromaDB中搜索
  const searchResults = await collection.query({
    queryEmbeddings: [queryVector],
    nResults: 3,
    include: ['documents', 'metadatas', 'distances']
  })
  
  console.log(`搜索结果数量: ${searchResults.documents[0].length}`)
  console.log('搜索结果:')
  
  for (let i = 0; i < searchResults.documents[0].length; i++) {
    console.log(`\n结果 ${i + 1}:`)
    console.log(`距离: ${searchResults.distances[0][i]}`)
    console.log(`文档: ${searchResults.documents[0][i].substring(0, 200)}...`)
    console.log(`元数据:`, searchResults.metadatas[0][i])
  }
  
  // 准备RAG功能
  console.log('\n🤖 正在初始化RAG问答系统...')
  
  // 初始化语言模型
  const llm = new ChatAlibabaTongyi({
    model: 'qwen-plus',
    streaming: true,  // 启用流式输出
    temperature: 0.1,
  })
  
  // 创建RAG提示模板
  const ragPrompt = ChatPromptTemplate.fromTemplate(`
基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请说明你无法基于提供的文档回答这个问题。

上下文信息：
{context}

用户问题：{question}

请提供准确、详细的回答：`)

  // 创建格式化上下文的函数
  const formatContext = (docs) => {
    return docs.map((doc, index) => `文档${index + 1}: ${doc}`).join('\n\n')
  }
  
  // 创建RAG链
  const ragChain = RunnableSequence.from([
    {
      context: () => formatContext(searchResults.documents[0]),
      question: new RunnablePassthrough(),
    },
    ragPrompt,
    llm,
    new StringOutputParser(),
  ])
  
  console.log('\n🎯 RAG系统已就绪！开始回答问题...\n')
  
  // 执行RAG查询并流式输出
  const stream = await ragChain.stream(query)
  
  console.log('💬 AI回答：')
  process.stdout.write('> ')
  
  for await (const chunk of stream) {
    process.stdout.write(chunk)
  }
  
  console.log('\n\n🎉 RAG查询完成！')
  
  // 创建交互式问答函数
  const askQuestion = async (question) => {
    console.log(`\n🔍 正在搜索相关文档...`)
    
    // 生成查询向量
    const queryVector = await embeddings.embedQuery(question)
    
    // 在ChromaDB中搜索
    const searchResults = await collection.query({
      queryEmbeddings: [queryVector],
      nResults: 3,
      include: ['documents', 'metadatas', 'distances']
    })
    
    console.log(`📄 找到 ${searchResults.documents[0].length} 个相关文档`)
    
    // 更新RAG链的上下文
    const ragChainForQuestion = RunnableSequence.from([
      {
        context: () => formatContext(searchResults.documents[0]),
        question: new RunnablePassthrough(),
      },
      ragPrompt,
      llm,
      new StringOutputParser(),
    ])
    
    // 执行RAG查询并流式输出
    const stream = await ragChainForQuestion.stream(question)
    
    console.log('\n💬 AI回答：')
    process.stdout.write('> ')
    
    for await (const chunk of stream) {
      process.stdout.write(chunk)
    }
    
    console.log('\n')
  }
  
  // 示例：回答更多问题
  console.log('\n📚 现在可以回答更多问题了！')
  
  // 可以添加更多问题示例
  const additionalQuestions = [
    '小程序的使用说明',
  ]
  
  for (const additionalQuestion of additionalQuestions) {
    console.log(`\n❓ 问题: ${additionalQuestion}`)
    await askQuestion(additionalQuestion)
    console.log('\n' + '='.repeat(80))
  }
  
} catch (error) {
  console.error('相似性搜索失败:', error.message)
  console.error('详细错误:', error)
  throw error
}

console.log('\n✅ RAG系统完全配置完成！可以处理任何关于PDF文档的问题！')