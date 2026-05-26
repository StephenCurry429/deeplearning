<!-- 模型训练页面 -->
<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useSentimentStore } from '@/stores/sentiment'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Check, Setting, Tools, MagicStick, VideoPlay, VideoPause, Monitor, TrendingUp, BarChart3 } from '@element-plus/icons-vue'

const sentimentStore = useSentimentStore()

// 响应式数据
const trainingForm = ref({
  model_type: 'textcnn',
  language: 'chinese',
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.001
})

const showAdvanced = ref(false)
const trainingHistory = ref<any[]>([])

// 训练指标数据
const trainingMetrics = ref({
  train_loss: [] as number[],
  val_loss: [] as number[],
  train_acc: [] as number[],
  val_acc: [] as number[]
})

// 当前训练轮次
const currentEpoch = ref(0)

// 图表容器引用
const chartContainer = ref<HTMLElement | null>(null)

// 模型配置
const modelConfigs = {
  textcnn: {
    name: 'TextCNN',
    description: '基于卷积神经网络的文本分类模型，适合短文本快速分析',
    icon: '🔍',
    pros: ['训练速度快', '适合短文本', '模型体积小'],
    cons: ['长文本效果一般', '无法捕获长距离依赖'],
    complexity: 'low',
    recommendedParams: {
      epochs: 10,
      batch_size: 64,
      learning_rate: 0.001
    }
  },
  bilstm: {
    name: 'BiLSTM',
    description: '基于双向长短期记忆网络的序列模型，能捕获上下文信息',
    icon: '🔄',
    pros: ['捕获序列信息', '处理变长文本', '效果稳定'],
    cons: ['训练时间较长', '内存消耗大'],
    complexity: 'medium',
    recommendedParams: {
      epochs: 15,
      batch_size: 32,
      learning_rate: 0.0005
    }
  },
  bert: {
    name: 'BERT',
    description: '基于Transformer的预训练语言模型，具有最佳性能',
    icon: '🧠',
    pros: ['效果最佳', '预训练模型', '支持多语言'],
    cons: ['计算资源需求高', '训练时间长', '模型体积大'],
    complexity: 'high',
    recommendedParams: {
      epochs: 5,
      batch_size: 16,
      learning_rate: 0.00002
    }
  }
}

const languageConfigs = {
  chinese: {
    name: '中文',
    dataset: 'ChnSentiCorp',
    icon: '🇨🇳',
    description: '使用中文情感分析数据集训练模型'
  },
  english: {
    name: 'English',
    dataset: 'IMDb',
    icon: '🇺🇸',
    description: '使用英文电影评论数据集训练模型'
  }
}

// 计算属性
const currentModel = computed(() => {
  return modelConfigs[trainingForm.value.model_type as keyof typeof modelConfigs]
})

const currentLanguage = computed(() => {
  return languageConfigs[trainingForm.value.language as keyof typeof languageConfigs]
})

const isTraining = computed(() => sentimentStore.isTraining)
const trainingProgress = computed(() => sentimentStore.trainingStatus.progress)
const trainingMessage = computed(() => sentimentStore.trainingStatus.message)
const trainingError = computed(() => sentimentStore.trainingStatus.error)

const estimatedTime = computed(() => {
  const baseTime = {
    textcnn: 5,
    bilstm: 15,
    bert: 30
  }
  const timePerEpoch = baseTime[trainingForm.value.model_type as keyof typeof baseTime]
  return timePerEpoch * trainingForm.value.epochs
})

// 应用推荐参数
const applyRecommended = () => {
  const recommended = currentModel.value.recommendedParams
  trainingForm.value.epochs = recommended.epochs
  trainingForm.value.batch_size = recommended.batch_size
  trainingForm.value.learning_rate = recommended.learning_rate
  ElMessage.success('已应用推荐参数')
}

// 开始训练
const startTraining = async () => {
  try {
    const confirmed = await ElMessageBox.confirm(
      `确定要开始训练 ${currentModel.value.name} 模型吗？\n` +
      `语言: ${currentLanguage.value.name}\n` +
      `预计训练时间: ${estimatedTime.value} 分钟`,
      '确认训练',
      {
        confirmButtonText: '开始训练',
        cancelButtonText: '取消',
        type: 'info'
      }
    )

    if (confirmed) {
      const success = await sentimentStore.trainModel(trainingForm.value)

      if (success) {
        ElMessage.success('模型训练已启动')
      }
    }
  } catch (error) {
    // 用户取消操作
  }
}

// 停止训练
const stopTraining = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要停止训练吗？当前进度将丢失。',
      '确认停止',
      {
        confirmButtonText: '停止训练',
        cancelButtonText: '继续训练',
        type: 'warning'
      }
    )
    ElMessage.warning('停止训练功能待实现')
  } catch (error) {
    // 用户取消操作
  }
}

// 获取复杂度颜色
const getComplexityColor = (complexity: string) => {
  const colors = {
    low: 'success',
    medium: 'warning',
    high: 'danger'
  }
  return colors[complexity as keyof typeof colors] || 'info'
}

// 获取复杂度文本
const getComplexityText = (complexity: string) => {
  const texts = {
    low: '简单',
    medium: '中等',
    high: '复杂'
  }
  return texts[complexity as keyof typeof texts] || '未知'
}

// 格式化训练时间
const formatDuration = (minutes: number) => {
  if (minutes < 60) {
    return `${minutes} 分钟`
  } else {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    return `${hours} 小时 ${mins} 分钟`
  }
}

// 获取数据集状态
const getDatasetStatus = (language: string) => {
  // 从后端API数据中查找对应语言的数据集状态
  const dataset = sentimentStore.datasetsInfo?.available_datasets?.find(
    (d: any) => d.language === language
  )

  if (dataset?.downloaded) {
    return { type: 'success', text: '已下载' }
  } else {
    return { type: 'warning', text: '将自动下载' }
  }
}

// 监听训练状态变化，更新指标
watch(
  () => sentimentStore.trainingStatus,
  (newStatus) => {
    if (newStatus.results && newStatus.results.epoch) {
      // 更新当前轮次
      currentEpoch.value = newStatus.results.epoch

      // 更新训练指标
      if (newStatus.results.train_loss !== undefined) {
        trainingMetrics.value.train_loss.push(newStatus.results.train_loss)
      }
      if (newStatus.results.val_loss !== undefined) {
        trainingMetrics.value.val_loss.push(newStatus.results.val_loss)
      }
      if (newStatus.results.train_acc !== undefined) {
        trainingMetrics.value.train_acc.push(newStatus.results.train_acc)
      }
      if (newStatus.results.val_acc !== undefined) {
        trainingMetrics.value.val_acc.push(newStatus.results.val_acc)
      }

      // 更新图表
      updateCharts()
    }
  },
  { deep: true }
)

// 更新图表
const updateCharts = () => {
  nextTick(() => {
    // 使用Canvas绘制简单图表
    drawLossChart()
    drawAccuracyChart()
  })
}

// 绘制损失图表
const drawLossChart = () => {
  const canvas = document.getElementById('lossChart') as HTMLCanvasElement
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const width = canvas.width
  const height = canvas.height
  const padding = 40

  // 清空画布
  ctx.clearRect(0, 0, width, height)

  // 获取数据
  const trainLoss = trainingMetrics.value.train_loss
  const valLoss = trainingMetrics.value.val_loss

  if (trainLoss.length === 0) return

  // 计算范围
  const allLosses = [...trainLoss, ...valLoss].filter((v) => v !== undefined)
  const minLoss = Math.min(...allLosses) * 0.9
  const maxLoss = Math.max(...allLosses) * 1.1
  const rangeLoss = maxLoss - minLoss || 1

  // 绘制网格
  ctx.strokeStyle = '#eee'
  ctx.lineWidth = 1
  for (let i = 0; i <= 5; i++) {
    const y = padding + (height - 2 * padding) * (i / 5)
    ctx.beginPath()
    ctx.moveTo(padding, y)
    ctx.lineTo(width - padding, y)
    ctx.stroke()

    // 标注Y轴
    const value = maxLoss - (rangeLoss * i) / 5
    ctx.fillStyle = '#999'
    ctx.font = '12px Arial'
    ctx.textAlign = 'right'
    ctx.fillText(value.toFixed(2), padding - 10, y + 4)
  }

  // 绘制X轴标注
  ctx.fillStyle = '#999'
  ctx.font = '12px Arial'
  ctx.textAlign = 'center'
  for (let i = 0; i < trainLoss.length; i++) {
    const x = padding + ((width - 2 * padding) / (trainLoss.length - 1 || 1)) * i
    ctx.fillText(`${i + 1}`, x, height - padding + 20)
  }

  // 绘制训练损失
  if (trainLoss.length > 0) {
    ctx.beginPath()
    ctx.strokeStyle = '#409eff'
    ctx.lineWidth = 2
    ctx.moveTo(padding, height - padding - ((trainLoss[0] - minLoss) / rangeLoss) * (height - 2 * padding))

    for (let i = 1; i < trainLoss.length; i++) {
      const x = padding + ((width - 2 * padding) / (trainLoss.length - 1)) * i
      const y = height - padding - ((trainLoss[i] - minLoss) / rangeLoss) * (height - 2 * padding)
      ctx.lineTo(x, y)
    }
    ctx.stroke()

    // 绘制点
    trainLoss.forEach((loss, i) => {
      const x = padding + ((width - 2 * padding) / (trainLoss.length - 1)) * i
      const y = height - padding - ((loss - minLoss) / rangeLoss) * (height - 2 * padding)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#409eff'
      ctx.fill()
    })
  }

  // 绘制验证损失
  if (valLoss.length > 0) {
    ctx.beginPath()
    ctx.strokeStyle = '#f56c6c'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.moveTo(padding, height - padding - ((valLoss[0] - minLoss) / rangeLoss) * (height - 2 * padding))

    for (let i = 1; i < valLoss.length; i++) {
      const x = padding + ((width - 2 * padding) / (valLoss.length - 1)) * i
      const y = height - padding - ((valLoss[i] - minLoss) / rangeLoss) * (height - 2 * padding)
      ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.setLineDash([])

    // 绘制点
    valLoss.forEach((loss, i) => {
      const x = padding + ((width - 2 * padding) / (valLoss.length - 1)) * i
      const y = height - padding - ((loss - minLoss) / rangeLoss) * (height - 2 * padding)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#f56c6c'
      ctx.fill()
    })
  }

  // 绘制图例
  ctx.fillStyle = '#409eff'
  ctx.fillRect(width - padding - 100, 20, 15, 3)
  ctx.fillStyle = '#666'
  ctx.font = '12px Arial'
  ctx.textAlign = 'left'
  ctx.fillText('训练损失', width - padding - 80, 23)

  ctx.fillStyle = '#f56c6c'
  ctx.fillRect(width - padding - 100, 40, 15, 3)
  ctx.setLineDash([5, 5])
  ctx.strokeRect(width - padding - 100, 40, 15, 3)
  ctx.setLineDash([])
  ctx.fillText('验证损失', width - padding - 80, 43)
}

// 绘制准确率图表
const drawAccuracyChart = () => {
  const canvas = document.getElementById('accuracyChart') as HTMLCanvasElement
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const width = canvas.width
  const height = canvas.height
  const padding = 40

  // 清空画布
  ctx.clearRect(0, 0, width, height)

  // 获取数据
  const trainAcc = trainingMetrics.value.train_acc
  const valAcc = trainingMetrics.value.val_acc

  if (trainAcc.length === 0) return

  // 绘制网格
  ctx.strokeStyle = '#eee'
  ctx.lineWidth = 1
  for (let i = 0; i <= 5; i++) {
    const y = padding + (height - 2 * padding) * (i / 5)
    ctx.beginPath()
    ctx.moveTo(padding, y)
    ctx.lineTo(width - padding, y)
    ctx.stroke()

    // 标注Y轴
    const value = ((5 - i) / 5) * 100
    ctx.fillStyle = '#999'
    ctx.font = '12px Arial'
    ctx.textAlign = 'right'
    ctx.fillText(`${Math.round(value)}%`, padding - 10, y + 4)
  }

  // 绘制X轴标注
  ctx.fillStyle = '#999'
  ctx.font = '12px Arial'
  ctx.textAlign = 'center'
  for (let i = 0; i < trainAcc.length; i++) {
    const x = padding + ((width - 2 * padding) / (trainAcc.length - 1 || 1)) * i
    ctx.fillText(`${i + 1}`, x, height - padding + 20)
  }

  // 绘制训练准确率
  if (trainAcc.length > 0) {
    ctx.beginPath()
    ctx.strokeStyle = '#67c23a'
    ctx.lineWidth = 2
    ctx.moveTo(padding, height - padding - (trainAcc[0] / 100) * (height - 2 * padding))

    for (let i = 1; i < trainAcc.length; i++) {
      const x = padding + ((width - 2 * padding) / (trainAcc.length - 1)) * i
      const y = height - padding - (trainAcc[i] / 100) * (height - 2 * padding)
      ctx.lineTo(x, y)
    }
    ctx.stroke()

    // 绘制点
    trainAcc.forEach((acc, i) => {
      const x = padding + ((width - 2 * padding) / (trainAcc.length - 1)) * i
      const y = height - padding - (acc / 100) * (height - 2 * padding)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#67c23a'
      ctx.fill()
    })
  }

  // 绘制验证准确率
  if (valAcc.length > 0) {
    ctx.beginPath()
    ctx.strokeStyle = '#e6a23c'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.moveTo(padding, height - padding - (valAcc[0] / 100) * (height - 2 * padding))

    for (let i = 1; i < valAcc.length; i++) {
      const x = padding + ((width - 2 * padding) / (valAcc.length - 1)) * i
      const y = height - padding - (valAcc[i] / 100) * (height - 2 * padding)
      ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.setLineDash([])

    // 绘制点
    valAcc.forEach((acc, i) => {
      const x = padding + ((width - 2 * padding) / (valAcc.length - 1)) * i
      const y = height - padding - (acc / 100) * (height - 2 * padding)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#e6a23c'
      ctx.fill()
    })
  }

  // 绘制图例
  ctx.fillStyle = '#67c23a'
  ctx.fillRect(width - padding - 100, 20, 15, 3)
  ctx.fillStyle = '#666'
  ctx.font = '12px Arial'
  ctx.textAlign = 'left'
  ctx.fillText('训练准确率', width - padding - 80, 23)

  ctx.fillStyle = '#e6a23c'
  ctx.fillRect(width - padding - 100, 40, 15, 3)
  ctx.setLineDash([5, 5])
  ctx.strokeRect(width - padding - 100, 40, 15, 3)
  ctx.setLineDash([])
  ctx.fillText('验证准确率', width - padding - 80, 43)
}

// 重置训练指标
const resetMetrics = () => {
  trainingMetrics.value = {
    train_loss: [],
    val_loss: [],
    train_acc: [],
    val_acc: []
  }
  currentEpoch.value = 0
  nextTick(() => {
    drawLossChart()
    drawAccuracyChart()
  })
}

// 监听开始训练，重置指标
watch(
  () => sentimentStore.isTraining,
  (isTraining) => {
    if (isTraining) {
      resetMetrics()
    }
  }
)

onMounted(() => {
  // 获取数据集信息
  sentimentStore.fetchDatasetsInfo()

  // 获取训练状态
  if (sentimentStore.isTraining) {
    resetMetrics()
  }

  // 初始化空图表
  nextTick(() => {
    drawLossChart()
    drawAccuracyChart()
  })
})
</script>

<template>
  <div class="training-page">
    <div class="page-header">
      <h1>
        <el-icon><Setting /></el-icon>
        模型训练
      </h1>
      <p>配置训练参数，训练自定义的情感分析模型</p>
    </div>

    <el-row :gutter="24">
      <!-- 训练配置 -->
      <el-col :lg="16" :md="24">
        <el-card class="training-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Tools /></el-icon>
              <span>训练配置</span>
              <el-button size="small" @click="applyRecommended">
                <el-icon><MagicStick /></el-icon>
                推荐参数
              </el-button>
            </div>
          </template>

          <el-form :model="trainingForm" label-width="120px" size="default">
            <!-- 模型选择 -->
            <el-form-item label="模型类型">
              <div class="model-cards-grid">
                <div
                  v-for="(config, key) in modelConfigs"
                  :key="key"
                  class="model-card-option"
                  :class="{ active: trainingForm.model_type === key }"
                  @click="trainingForm.model_type = key"
                >
                  <div class="model-card-content">
                    <div class="model-card-header">
                      <span class="model-icon">{{ config.icon }}</span>
                      <strong>{{ config.name }}</strong>
                      <el-tag
                        :type="getComplexityColor(config.complexity)"
                        size="small"
                        class="complexity-tag"
                      >
                        {{ getComplexityText(config.complexity) }}
                      </el-tag>
                    </div>
                    <div class="model-card-desc">{{ config.description }}</div>
                    <div class="model-card-features">
                      <div class="feature-section">
                        <span class="feature-label">优点:</span>
                        <ul class="feature-list">
                          <li v-for="pro in config.pros" :key="pro">{{ pro }}</li>
                        </ul>
                      </div>
                      <div class="feature-section">
                        <span class="feature-label">缺点:</span>
                        <ul class="feature-list cons">
                          <li v-for="con in config.cons" :key="con">{{ con }}</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                  <div class="selection-indicator">
                    <el-icon v-if="trainingForm.model_type === key"><Check /></el-icon>
                  </div>
                </div>
              </div>
            </el-form-item>

            <!-- 语言选择 -->
            <el-form-item label="训练语言">
              <el-radio-group v-model="trainingForm.language">
                <el-radio
                  v-for="(config, key) in languageConfigs"
                  :key="key"
                  :value="key"
                >
                  <span class="language-icon">{{ config.icon }}</span>
                  {{ config.name }} ({{ config.dataset }})
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <!-- 基础参数 -->
            <el-form-item label="训练轮数">
              <el-input-number
                v-model="trainingForm.epochs"
                :min="1"
                :max="50"
                controls-position="right"
                style="width: 150px;"
              />
              <span class="param-tip">建议: {{ currentModel.recommendedParams.epochs }} 轮</span>
            </el-form-item>

            <!-- 高级参数 -->
            <el-collapse v-model="showAdvanced" class="advanced-params">
              <el-collapse-item title="高级参数" name="advanced">
                <el-form-item label="批次大小">
                  <el-input-number
                    v-model="trainingForm.batch_size"
                    :min="8"
                    :max="128"
                    :step="8"
                    controls-position="right"
                    style="width: 150px;"
                  />
                  <span class="param-tip">建议: {{ currentModel.recommendedParams.batch_size }}</span>
                </el-form-item>

                <el-form-item label="学习率">
                  <el-input-number
                    v-model="trainingForm.learning_rate"
                    :min="0.00001"
                    :max="0.01"
                    :step="0.00001"
                    :precision="5"
                    controls-position="right"
                    style="width: 150px;"
                  />
                  <span class="param-tip">建议: {{ currentModel.recommendedParams.learning_rate }}</span>
                </el-form-item>
              </el-collapse-item>
            </el-collapse>
          </el-form>

          <!-- 训练控制 -->
          <div class="training-controls">
            <el-button
              v-if="!isTraining"
              type="primary"
              size="large"
              @click="startTraining"
            >
              <el-icon><VideoPlay /></el-icon>
              开始训练
            </el-button>
            <el-button
              v-else
              type="danger"
              size="large"
              @click="stopTraining"
            >
              <el-icon><VideoPause /></el-icon>
              停止训练
            </el-button>
          </div>

          <!-- 训练进度 -->
          <div v-if="isTraining" class="training-progress">
            <h4>训练进度</h4>
            <el-progress
              :percentage="trainingProgress"
              :format="(percentage: number) => `${percentage}%`"
              :stroke-width="12"
              :status="trainingError ? 'exception' : 'active'"
            />
            <div class="progress-text">
              {{ trainingMessage || '正在训练...' }}
            </div>
            <div v-if="trainingError" class="error-text">
              错误: {{ trainingError }}
            </div>
          </div>

          <!-- 训练可视化图表 -->
          <div class="training-visualization">
            <h4 class="visualization-header">
              <el-icon><TrendingUp /></el-icon>
              训练指标可视化
            </h4>

            <!-- 当前指标 -->
            <div class="current-metrics">
              <div class="metric-card">
                <div class="metric-label">当前轮次</div>
                <div class="metric-value">{{ currentEpoch }} / {{ trainingForm.epochs }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">训练损失</div>
                <div class="metric-value loss">
                  {{ trainingMetrics.train_loss.length > 0 
                    ? trainingMetrics.train_loss[trainingMetrics.train_loss.length - 1].toFixed(4) 
                    : '-' 
                  }}
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">训练准确率</div>
                <div class="metric-value accuracy">
                  {{ trainingMetrics.train_acc.length > 0 
                    ? `${trainingMetrics.train_acc[trainingMetrics.train_acc.length - 1].toFixed(1)}%` 
                    : '-' 
                  }}
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">验证准确率</div>
                <div class="metric-value val-accuracy">
                  {{ trainingMetrics.val_acc.length > 0 
                    ? `${trainingMetrics.val_acc[trainingMetrics.val_acc.length - 1].toFixed(1)}%` 
                    : '-' 
                  }}
                </div>
              </div>
            </div>

            <!-- 损失图表 -->
            <div class="chart-section">
              <h5><el-icon><BarChart3 /></el-icon> 损失曲线</h5>
              <canvas id="lossChart" width="600" height="250" class="training-chart"></canvas>
            </div>

            <!-- 准确率图表 -->
            <div class="chart-section">
              <h5><el-icon><TrendingUp /></el-icon> 准确率曲线</h5>
              <canvas id="accuracyChart" width="600" height="250" class="training-chart"></canvas>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 训练状态 -->
      <el-col :lg="8" :md="24">
        <el-card class="status-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Monitor /></el-icon>
              <span>训练状态</span>
            </div>
          </template>

          <!-- 当前状态 -->
          <div class="current-status">
            <div class="status-item">
              <div class="status-label">训练状态</div>
              <el-tag
                :type="isTraining ? 'warning' : 'success'"
                size="large"
              >
                {{ isTraining ? '训练中' : '空闲' }}
              </el-tag>
            </div>

            <div class="status-item">
              <div class="status-label">选中模型</div>
              <div class="status-value">
                <span class="model-icon">{{ currentModel.icon }}</span>
                {{ currentModel.name }}
              </div>
            </div>

            <div class="status-item">
              <div class="status-label">训练语言</div>
              <div class="status-value">
                <span class="language-icon">{{ currentLanguage.icon }}</span>
                {{ currentLanguage.name }}
              </div>
            </div>
          </div>

          <!-- 数据集状态 -->
          <div class="dataset-status">
            <h4>数据集状态</h4>
            <div class="status-item">
              <div class="status-label">{{ currentLanguage.name }}数据集</div>
              <div class="status-value">
                <el-tag
                  :type="getDatasetStatus(trainingForm.language).type"
                  size="small"
                >
                  {{ getDatasetStatus(trainingForm.language).text }}
                </el-tag>
              </div>
            </div>
          </div>

          <!-- 使用提示 -->
          <div class="usage-tips">
            <h4>训练建议</h4>
            <el-alert
              title="智能训练提示"
              type="info"
              :closable="false"
              show-icon
            >
              <template #default>
                <ul>
                  <li>系统会自动检测并使用已下载的数据集</li>
                  <li>如果数据集未下载，将自动下载（支持多数据源）</li>
                  <li>首次训练建议使用推荐参数</li>
                  <li>BERT模型需要较长训练时间</li>
                  <li>可通过调整批次大小优化内存使用</li>
                </ul>
              </template>
            </el-alert>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.training-page {
  width: 100%;
  padding: 0 20px;
  min-height: 100vh;
  overflow-x: hidden;
}

.page-header {
  text-align: center;
  margin-bottom: 30px;
}

.page-header h1 {
  font-size: 28px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.page-header p {
  color: #606266;
  font-size: 16px;
  margin: 0;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.card-header span {
  font-weight: 600;
}

.model-cards-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  width: 100%;
}

.model-card-option {
  position: relative;
  border: 2px solid #ebeef5;
  border-radius: 12px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #ffffff;
  min-height: 280px;
}

.model-card-option:hover {
  border-color: #409eff;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.15);
  transform: translateY(-2px);
}

.model-card-option.active {
  border-color: #409eff;
  background: linear-gradient(135deg, #f8fbff 0%, #e6f4ff 100%);
  box-shadow: 0 4px 16px rgba(64, 158, 255, 0.2);
}

.model-card-content {
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.model-icon {
  font-size: 24px;
}

.complexity-tag {
  margin-left: auto;
}

.model-card-desc {
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
  margin-bottom: 12px;
}

.model-card-features {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.feature-section {
  flex: 1;
}

.feature-label {
  font-size: 12px;
  font-weight: 600;
  color: #303133;
  display: block;
  margin-bottom: 6px;
}

.feature-list {
  margin: 0;
  padding-left: 16px;
  font-size: 12px;
  line-height: 1.4;
}

.feature-list li {
  margin-bottom: 4px;
  color: #67c23a;
}

.feature-list.cons li {
  color: #f56c6c;
}

.selection-indicator {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #409eff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: all 0.3s ease;
}

.model-card-option.active .selection-indicator {
  opacity: 1;
}

.language-icon {
  font-size: 16px;
  margin-right: 4px;
}

.param-tip {
  margin-left: 12px;
  font-size: 12px;
  color: #909399;
}

.advanced-params {
  margin: 20px 0;
}

.training-controls {
  text-align: center;
  margin: 30px 0;
}

.training-progress {
  margin-top: 30px;
}

.training-progress h4 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
}

.progress-text {
  text-align: center;
  color: #606266;
  font-size: 14px;
  margin-top: 8px;
}

.error-text {
  text-align: center;
  color: #f56c6c;
  font-size: 14px;
  margin-top: 8px;
}

/* 训练可视化样式 */
.training-visualization {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}

.visualization-header {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.current-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.metric-card {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  transition: all 0.3s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
}

.metric-value.loss {
  color: #f56c6c;
}

.metric-value.accuracy {
  color: #67c23a;
}

.metric-value.val-accuracy {
  color: #e6a23c;
}

.chart-section {
  margin-bottom: 24px;
}

.chart-section h5 {
  font-size: 14px;
  font-weight: 600;
  color: #606266;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.training-chart {
  width: 100%;
  max-width: 600px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #ebeef5;
}

.current-status {
  margin-bottom: 30px;
}

.dataset-status {
  margin-bottom: 30px;
}

.dataset-status h4 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.status-item:last-child {
  border-bottom: none;
}

.status-label {
  font-weight: 600;
  color: #303133;
}

.status-value {
  font-size: 14px;
  color: #606266;
  text-align: right;
}

.usage-tips {
  margin-top: 30px;
}

.usage-tips h4 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
}

.usage-tips ul {
  margin: 0;
  padding-left: 20px;
}

.usage-tips li {
  margin-bottom: 8px;
  color: #606266;
  font-size: 14px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .model-cards-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }

  .model-card-option {
    min-height: 260px;
    padding: 16px;
  }
}

@media (max-width: 768px) {
  .training-page {
    padding: 0 10px;
  }

  .model-cards-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .model-card-option {
    min-height: 240px;
    padding: 16px;
  }

  .model-card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .complexity-tag {
    margin-left: 0;
    align-self: flex-start;
  }

  .status-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .status-value {
    text-align: left;
  }

  .selection-indicator {
    top: 8px;
    right: 8px;
    width: 20px;
    height: 20px;
  }
}
</style>
