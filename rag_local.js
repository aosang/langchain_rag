import path from 'path'
import 'dotenv/config'
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { AlibabaTongyiEmbeddings } from '@langchain/community/embeddings/alibaba_tongyi'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'

const pdfPath = path.resolve('file/rag_test.pdf')
const loader = new PDFLoader(pdfPath)

const docs = await loader.load()

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})

const allSplits = await textSplitter.splitDocuments(docs)

const embeddings = new AlibabaTongyiEmbeddings({})
const vectorStore = new MemoryVectorStore(embeddings)
await vectorStore.addDocuments(allSplits)

const results = await vectorStore.similaritySearch('When was Nike incorporated?')
console.log('results:', results.length, results[0])