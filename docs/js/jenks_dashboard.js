// 3級Jenks分級分析儀表板 JavaScript
// 處理圖表渲染、交互功能和數據展示

let scatterChart = null;
let radarChart = null;

// 初始化儀表板
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 初始化3級Jenks分級儀表板...');
    
    // 等待一段時間確保 DASHBOARD_DATA 已載入
    setTimeout(() => {
        if (typeof DASHBOARD_DATA !== 'undefined' && DASHBOARD_DATA.scatter_data && DASHBOARD_DATA.radar_data && DASHBOARD_DATA.level_statistics) {
            console.log('DASHBOARD_DATA 已載入:', DASHBOARD_DATA);
            initializeDashboard();
        } else {
            console.error('❌ 無法載入儀表板數據 (DASHBOARD_DATA 未定義或不完整)');
            if (typeof DASHBOARD_DATA !== 'undefined') {
                console.log('DASHBOARD_DATA 結構:', DASHBOARD_DATA);
            } else {
                console.log('DASHBOARD_DATA 未定義');
            }
            // Display error message to user
            const sections = document.querySelectorAll('.metrics-cards, .chart-section, .data-table-section');
            sections.forEach(section => {
                section.innerHTML = '<p style="color: red; text-align: center;">儀表板數據載入失敗，請檢查數據文件或聯繫管理員。</p>';
            });
        }
    }, 250); // 增加延遲確保 jenks_data.js 載入
});

function initializeDashboard() {
    console.log('開始初始化儀表板元件...');
    
    updateMetricCards();
    initializeScatterChart();
    initializeRadarChart();
    populateDataTable();
    setupDistrictSelector();
    setupCardInteractions();
    loadChartStatistics(); // This function might need DASHBOARD_DATA.summary or map_statistics.json
    setupAnchorNavigation(); // 初始化錨點導航
    
    // 頁面載入後滾動到概覽區域（居中顯示）
    setTimeout(() => {
        const overviewSection = document.getElementById('overview');
        if (overviewSection) {
            // 計算更精確的居中位置
            const rect = overviewSection.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            const elementHeight = rect.height;
            const targetScroll = window.pageYOffset + rect.top - (windowHeight - elementHeight) / 2;
            
            window.scrollTo({
                top: Math.max(0, targetScroll),
                behavior: 'smooth'
            });
        }
    }, 300);  // 增加延遲確保所有元素載入完成
    
    console.log('儀表板初始化完成');
    
    // 備用滾動邏輯：等待圖表完全渲染後再滾動
    setTimeout(() => {
        scrollToCenter();
    }, 800);
}

// 專用的居中滾動函數
function scrollToCenter() {
    const overviewSection = document.getElementById('overview');
    if (overviewSection) {
        // 等待所有內容載入完成後再計算位置
        requestAnimationFrame(() => {
            const rect = overviewSection.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            const currentScroll = window.pageYOffset;
            const sectionHeight = rect.height;
            
            // 計算讓section在視窗中央的位置
            const targetScroll = currentScroll + rect.top - (windowHeight - sectionHeight) / 2;
            
            // 確保不滾動到負數位置
            const finalScroll = Math.max(0, targetScroll);
            
            console.log('滾動計算:', {
                currentScroll,
                rectTop: rect.top,
                windowHeight,
                sectionHeight,
                targetScroll,
                finalScroll
            });
            
            window.scrollTo({
                top: finalScroll,
                behavior: 'smooth'
            });
            
            console.log('已滾動到概覽區域中心位置');
        });
    }
}

// 設置卡片互動功能
function setupCardInteractions() {
    console.log('設置卡片互動功能...');
    
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.addEventListener('click', function() {
            const cardType = ['high', 'medium', 'low'][index];
            const levelName = ['高潛力', '中潛力', '低潛力'][index];
            
            // 高亮表格中對應的行
            highlightTableRows(levelName);
            
            // 滾動到表格
            const tableSection = document.querySelector('.data-table-section');
            if (tableSection) {
                tableSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            // 添加視覺反饋
            card.style.transform = 'translateY(-16px) scale(1.08)';
            setTimeout(() => {
                card.style.transform = '';
            }, 200);
            
            console.log(`點擊了${levelName}卡片，已篩選表格`);
        });
        
        // 移除圖標相關的hover效果，因為已經不使用圖標了
    });
}

// 高亮表格中對應潛力等級的行
function highlightTableRows(levelName) {
    const tableRows = document.querySelectorAll('#dataTable tbody tr');
    
    // 清除之前的高亮
    tableRows.forEach(row => {
        row.classList.remove('highlighted-row');
        row.style.backgroundColor = '';
    });
    
    // 添加新的高亮
    setTimeout(() => {
        tableRows.forEach(row => {
            const levelBadge = row.querySelector('.level-badge');
            if (levelBadge && levelBadge.textContent.trim() === levelName) {
                row.classList.add('highlighted-row');
                row.style.backgroundColor = 'rgba(255, 255, 255, 0.15)';
                row.style.transition = 'background-color 0.3s ease';
            }
        });
        
        // 3秒後移除高亮
        setTimeout(() => {
            tableRows.forEach(row => {
                row.classList.remove('highlighted-row');
                row.style.backgroundColor = '';
            });
        }, 3000);
    }, 100);
}

function updateMetricCards() {
    console.log('📋 更新指標卡片...');
    if (!DASHBOARD_DATA || !DASHBOARD_DATA.level_statistics) {
        console.error('更新指標卡片失敗: DASHBOARD_DATA.level_statistics 未定義');
        return;
    }

    const stats = DASHBOARD_DATA.level_statistics;
    
    const highCountEl = document.getElementById('highCount'); // Using ID from HTML
    const highDistrictsEl = document.getElementById('highDistricts'); // Using ID from HTML
    if (highCountEl) highCountEl.textContent = stats.high_potential.count;
    if (highDistrictsEl) highDistrictsEl.textContent = stats.high_potential.districts.join(' ') || '無';  // 改為空格分隔
    
    const mediumCountEl = document.getElementById('mediumCount'); // Using ID from HTML
    const mediumDistrictsEl = document.getElementById('mediumDistricts'); // Using ID from HTML
    if (mediumCountEl) mediumCountEl.textContent = stats.medium_potential.count;
    if (mediumDistrictsEl) mediumDistrictsEl.textContent = stats.medium_potential.districts.join(' ') || '無';  // 改為空格分隔
    
    const lowCountEl = document.getElementById('lowCount'); // Using ID from HTML
    const lowDistrictsEl = document.getElementById('lowDistricts'); // Using ID from HTML
    if (lowCountEl) lowCountEl.textContent = stats.low_potential.count;
    if (lowDistrictsEl) lowDistrictsEl.textContent = stats.low_potential.districts.join(' ') || '無';  // 改為空格分隔
}

// 自定義插件：背景區域
const backgroundRegionsPlugin = {
    id: 'backgroundRegions',
    beforeDraw: (chart) => {
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        const yScale = chart.scales.y;
        const xScale = chart.scales.x;
        
        // 獲取分級閾值（需要根據實際數據計算）
        const highThreshold = 6.7;  // 高潛力閾值
        const mediumThreshold = 3.5; // 中潛力閾值
        
        const regions = [
            { min: highThreshold, max: yScale.max, color: 'rgba(235, 112, 98, 0.08)', label: '高潛力區' },
            { min: mediumThreshold, max: highThreshold, color: 'rgba(245, 176, 65, 0.08)', label: '中潛力區' },
            { min: yScale.min, max: mediumThreshold, color: 'rgba(92, 172, 226, 0.08)', label: '低潛力區' }
        ];
        
        ctx.save();
        regions.forEach(region => {
            const yTop = yScale.getPixelForValue(region.max);
            const yBottom = yScale.getPixelForValue(region.min);
            
            ctx.fillStyle = region.color;
            ctx.fillRect(chartArea.left, yTop, chartArea.width, yBottom - yTop);
        });
        ctx.restore();
    }
};

function initializeScatterChart() {
    console.log('初始化散點圖...');
    if (!DASHBOARD_DATA || !DASHBOARD_DATA.scatter_data) {
        console.error('初始化散點圖失敗: DASHBOARD_DATA.scatter_data 未定義');
        return;
    }
    
    const ctx = document.getElementById('jenksChart'); // Corrected ID from HTML
    if (!ctx) {
        console.error('找不到散點圖 canvas 元素 (ID: jenksChart)');
        return;
    }
    
    const canvasContext = ctx.getContext('2d');
    const scatterPlotData = DASHBOARD_DATA.scatter_data; // Renamed to avoid confusion with D3/other scatter
    
    const groupedData = {
        '高潛力': [],
        '中潛力': [],
        '低潛力': []
    };
    
    // scatter_data from Python is already structured with x, y, name, level, color
    // We need to map 'level' (e.g., '高潛力') to the keys of groupedData
    scatterPlotData.forEach(point => {
        if (groupedData[point.level]) { // point.level is '高潛力', '中潛力', or '低潛力'
            groupedData[point.level].push({
                x: point.x, // rank
                y: point.y, // score
                label: point.name, // district name
            });
        } else {
            console.warn(`未知潛力等級用於散點圖: ${point.level} for ${point.name}`);
        }
    });
    
    // 使用數據中的正確顏色
    const levelColors = {
        '高潛力': '#eb7062',  // 紅色 - 高潛力
        '中潛力': '#f5b041',  // 橙色 - 中潛力  
        '低潛力': '#5cace2'   // 藍色 - 低潛力
    };
    
    const datasets = Object.keys(groupedData).map(levelKey => ({
        label: levelKey, // '高潛力', '中潛力', '低潛力'
        data: groupedData[levelKey],
        backgroundColor: levelColors[levelKey],
        borderColor: levelColors[levelKey],
        pointRadius: 2,
        pointHoverRadius: 4,
        pointBorderWidth: 0, // 移除邊框
        showLine: false // 確保不顯示連線
    }));
    
    if (scatterChart) {
        scatterChart.destroy();
    }

    const maxXValue = Math.max(...scatterPlotData.map(p => p.x), 0) + 1; // rank + 1 for padding
    const maxYValue = Math.max(...scatterPlotData.map(p => p.y), 0);

    scatterChart = new Chart(canvasContext, {
        type: 'scatter',
        data: { datasets: datasets },
        plugins: [ChartDataLabels, backgroundRegionsPlugin],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: false  // 移除圖表標題
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: { 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff', 
                        usePointStyle: true, 
                        padding: 20, 
                        font: { size: 12 },
                        pointStyle: 'circle',
                        boxWidth: 8,
                        boxHeight: 8
                    }
                },
                tooltip: {
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--surface-color') || '#1e1e1e',
                    titleColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
                    bodyColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
                    borderColor: getComputedStyle(document.documentElement).getPropertyValue('--border-color') || '#333333',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        title: function(context) {
                            const point = context[0].raw;
                            return `${point.label} (排名第${point.x}名)`;
                        },
                        label: function(context) {
                            const point = context.raw;
                            const dataset = context.dataset;
                            return [
                                `綜合分數: ${point.y.toFixed(1)}分`,
                                `潛力等級: ${dataset.label}`,
                                `點擊查看雷達圖分析 →`
                            ];
                        }
                    }
                },
                datalabels: {
                    display: true,
                    color: '#ffffff',
                    backgroundColor: 'transparent',  // 去掉背景框
                    borderWidth: 0,                  // 去掉邊框
                    font: {
                        size: 14,                    // 增大字體讓標籤更明顯
                        weight: '600'
                    },
                    formatter: function(value, context) {
                        return value.label.replace('區', '');
                    },
                    anchor: 'center',                // 錨點設定為圓點中心
                    align: 'top',                    // 對齊方式設為上方
                    offset: 6                        // 增加偏移量讓文字與圓點距離更遠
                }
            },
            scales: {
                x: {
                    title: { 
                        display: true, text: '排名', 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff', 
                        font: { size: 14, weight: '600' } 
                    },
                    ticks: { 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') || '#b3b3b3', 
                        font: { size: 12 } 
                    },
                    grid: { color: '#666666', lineWidth: 1 }, // 增強網格線可見度
                    min: 0,
                    max: maxXValue 
                },
                y: {
                    title: { 
                        display: true, text: '綜合分數 (0-10)', 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff', 
                        font: { size: 14, weight: '600' } 
                    },
                    ticks: { 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') || '#b3b3b3', 
                        font: { size: 12 } 
                    },
                    grid: { color: '#666666', lineWidth: 1 }, // 增強網格線可見度
                    min: 0,
                    max: Math.ceil(maxYValue / 2) * 2 + 2 // e.g. if max score is 9.4, max y is 12
                }
            },
            animation: { duration: 1200, easing: 'easeOutQuart' },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const element = elements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataIndex = element.index;
                    const clickedPoint = scatterChart.data.datasets[datasetIndex].data[dataIndex];
                    const districtName = clickedPoint.label;
                    
                    // 自動選擇雷達圖
                    const selector = document.getElementById('districtSelect');
                    if (selector) {
                        selector.value = districtName;
                        updateRadarChart(districtName);
                        
                        // 滾動到雷達圖區域
                        const radarSection = document.querySelector('.chart-section:has(#radarChart)');
                        if (radarSection) {
                            radarSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    }
                }
            }
        }
    });
    console.log("散點圖已初始化，圓點大小設為:", datasets[0]?.pointRadius || "未設定");
    console.log("所有數據集的圓點大小:", datasets.map(d => `${d.label}: ${d.pointRadius}`));
}

function initializeRadarChart() {
    console.log('初始化雷達圖...');
    // DASHBOARD_DATA.radar_data is an object where keys are district names
    // Example: DASHBOARD_DATA.radar_data['桃園區'].features
    // We need a sample district to get the feature labels for initialization
    const districtNames = DASHBOARD_DATA.districts ? DASHBOARD_DATA.districts.map(d => d.name) : [];
    if (districtNames.length === 0 || !DASHBOARD_DATA.radar_data || !DASHBOARD_DATA.radar_data[districtNames[0]]) {
        console.error('初始化雷達圖失敗: DASHBOARD_DATA.radar_data 或其 features 未定義 (基於第一個行政區)');
        return;
    }

    const ctx = document.getElementById('radarChart');
    if (!ctx) {
        console.error('找不到雷達圖 canvas 元素 (ID: radarChart)');
        return;
    }
    
    const canvasContext = ctx.getContext('2d');
    // Get features from the first district in radar_data
    const sampleDistrictName = Object.keys(DASHBOARD_DATA.radar_data)[0];
    const radarFeatureLabels = DASHBOARD_DATA.radar_data[sampleDistrictName]?.features || [];

    // Feature mapping for Chinese labels (if necessary, otherwise use Python provided ones)
    // Python already provides Chinese feature names in radar_data[district].features
    // So, radarFeatureLabels should already be in Chinese.

    if (radarChart) {
        radarChart.destroy();
    }
    radarChart = new Chart(canvasContext, {
        type: 'radar',
        data: {
            labels: radarFeatureLabels, // These should be Chinese from Python
            datasets: [] 
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '請選擇行政區查看特徵分析',
                    color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
                    font: { size: 14, weight: '600' }
                },
                legend: {
                    display: false, 
                    position: 'top',
                    labels: { 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff', 
                        font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--surface-color') || '#1e1e1e',
                    titleColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
                    bodyColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
                    borderColor: getComputedStyle(document.documentElement).getPropertyValue('--border-color') || '#333333',
                    borderWidth: 1
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    min: 0,
                    max: 100, // Normalized 0-100 scale from Python
                    ticks: { 
                        stepSize: 20, 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') || '#b3b3b3', 
                        backdropColor: 'transparent', 
                        font: { size: 10 } 
                    },
                    grid: { color: '#666666', lineWidth: 1 },
                    angleLines: { color: '#666666', lineWidth: 1 },
                    pointLabels: { 
                        color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff', 
                        font: { size: 12, weight: '600' } 
                    }
                }
            }
        }
    });
    console.log("雷達圖已初始化。");
}

function setupDistrictSelector() {
    console.log('設置區域選擇器...');
    if (!DASHBOARD_DATA || !DASHBOARD_DATA.districts) { // Use DASHBOARD_DATA.districts for selector
        console.error('設置區域選擇器失敗: DASHBOARD_DATA.districts 未定義');
        return;
    }

    const selector = document.getElementById('districtSelect');
    if (!selector) {
        console.error('找不到區域選擇器元素 (ID: districtSelect)');
        return;
    }
    
    selector.innerHTML = '<option value="">選擇行政區</option>';
    // DASHBOARD_DATA.districts contains { name, rank, score, level, level_en, features }
    // Sort by rank or score for consistency
    const sortedDistricts = [...DASHBOARD_DATA.districts].sort((a, b) => a.rank - b.rank); 
    
    sortedDistricts.forEach(district => {
        const option = document.createElement('option');
        option.value = district.name; // Use district.name (which is '區域別')
        option.textContent = `${district.name} (${district.level}, ${district.score.toFixed(1)}分)`;
        selector.appendChild(option);
    });
    
    selector.addEventListener('change', function() {
        if (this.value) {
            updateRadarChart(this.value);
        } else {
            clearRadarChart();
        }
    });
    console.log("區域選擇器已設置。");
}

function updateRadarChart(districtName) {
    console.log(`更新雷達圖: ${districtName}`);
    if (!DASHBOARD_DATA || !DASHBOARD_DATA.radar_data || !DASHBOARD_DATA.radar_data[districtName]) {
        console.error(`更新雷達圖失敗: DASHBOARD_DATA.radar_data['${districtName}'] 不完整`);
        clearRadarChart();
        return;
    }
    
    const districtRadarInfo = DASHBOARD_DATA.radar_data[districtName];
    const districtDisplayInfo = DASHBOARD_DATA.districts.find(d => d.name === districtName);
    const level = districtDisplayInfo ? districtDisplayInfo.level : 'N/A';

    // 計算平均值線數據
    const allDistricts = Object.values(DASHBOARD_DATA.radar_data);
    const averageValues = districtRadarInfo.features.map((feature, index) => {
        const sum = allDistricts.reduce((acc, district) => acc + district.values[index], 0);
        return sum / allDistricts.length;
    });

    const levelColors = {
        '高潛力': DASHBOARD_DATA.scatter_data.find(p => p.level === '高潛力')?.color || '#eb7062',
        '中潛力': DASHBOARD_DATA.scatter_data.find(p => p.level === '中潛力')?.color || '#f5b041',
        '低潛力': DASHBOARD_DATA.scatter_data.find(p => p.level === '低潛力')?.color || '#5cace2',
        'N/A': '#757575'
    };
    const radarColor = levelColors[level] || '#eb7062';
    
    radarChart.data.labels = districtRadarInfo.features;
    radarChart.data.datasets = [
        // 平均值線（背景）
        {
            label: '全市平均值',
            data: averageValues,
            backgroundColor: 'rgba(150, 150, 150, 0.1)',
            borderColor: 'rgba(150, 150, 150, 0.5)',
            pointBackgroundColor: 'rgba(150, 150, 150, 0.3)',
            pointBorderColor: 'rgba(150, 150, 150, 0.6)',
            pointBorderWidth: 1,
            pointRadius: 3,
            pointHoverRadius: 5,
            borderWidth: 1.5,
            borderDash: [5, 5]
        },
        // 選定區域數據
        {
            label: `${districtName} (${level})`,
            data: districtRadarInfo.values,
            backgroundColor: `${radarColor}4D`,
            borderColor: radarColor,
            pointBackgroundColor: radarColor,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
            borderWidth: 3
        }
    ];
    
    radarChart.options.plugins.title.text = `${districtName} 特徵分析 vs 全市平均 (標準化0-100)`;
    radarChart.options.plugins.legend.display = true;
    radarChart.options.plugins.tooltip = {
        backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--surface-color') || '#1e1e1e',
        titleColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
        bodyColor: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') || '#ffffff',
        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--border-color') || '#333333',
        borderWidth: 1,
        cornerRadius: 8,
        padding: 12,
        callbacks: {
            title: function(context) {
                return context[0].label;
            },
            label: function(context) {
                const value = context.parsed.r.toFixed(1);
                const avg = averageValues[context.dataIndex].toFixed(1);
                const diff = (context.parsed.r - averageValues[context.dataIndex]).toFixed(1);
                const rank = calculateFeatureRank(districtName, context.dataIndex);
                
                if (context.datasetIndex === 0) {
                    return `全市平均: ${avg}`;
                } else {
                    return [
                        `${districtName}: ${value}`,
                        `全市平均: ${avg}`,
                        `差異: ${diff > 0 ? '+' : ''}${diff}`,
                        `排名: 第${rank}名`
                    ];
                }
            }
        }
    };
    
    radarChart.update();
    console.log(`雷達圖已更新為 ${districtName} (含平均值對比)`);
}

// 計算特定特徵的排名
function calculateFeatureRank(districtName, featureIndex) {
    const allDistricts = Object.keys(DASHBOARD_DATA.radar_data);
    const values = allDistricts.map(name => ({
        name: name,
        value: DASHBOARD_DATA.radar_data[name].values[featureIndex]
    }));
    
    values.sort((a, b) => b.value - a.value);
    const rank = values.findIndex(item => item.name === districtName) + 1;
    return rank;
}

function clearRadarChart() {
    if(radarChart) {
        const sampleDistrictName = Object.keys(DASHBOARD_DATA.radar_data)[0];
        const radarFeatureLabels = DASHBOARD_DATA.radar_data[sampleDistrictName]?.features || [];
        radarChart.data.labels = radarFeatureLabels; // Reset labels
        radarChart.data.datasets = [];
        radarChart.options.plugins.title.text = '請選擇行政區查看特徵分析';
        radarChart.options.plugins.legend.display = false;
        radarChart.update();
        console.log("雷達圖已清除。");
    }
}

function populateDataTable() { // If re-enabled, use DASHBOARD_DATA.districts
    console.log('📝 填充數據表格...');
    if (!DASHBOARD_DATA || !DASHBOARD_DATA.districts) {
        console.error('填充數據表格失敗: DASHBOARD_DATA.districts 未定義');
        return;
    }

    const tableBody = document.getElementById('tableBody'); // ID from HTML
    if (!tableBody) {
        console.error('找不到表格 tbody 元素 (ID: tableBody)');
        return;
    }
    
    // DASHBOARD_DATA.districts = [{ name, rank, score, level, level_en, features: { '中文特徵名': value } }, ...]
    // Sort by rank
    const sortedData = [...DASHBOARD_DATA.districts].sort((a, b) => a.rank - b.rank);
    tableBody.innerHTML = ''; 
    
    // Define feature keys based on the order in table headers in index.html
    // These need to exactly match the keys in district.features which are Chinese from Python
    // Example: '工作年齡人口比例', '家戶中位數所得', '第三產業比例', '醫療指數', '商業集中度指數'
    // We can get the expected feature names from the first district's features keys if they are consistent
    const featureOrder = DASHBOARD_DATA.districts[0] ? Object.keys(DASHBOARD_DATA.districts[0].features) : [];
    // Or define them explicitly if the order from python isn't guaranteed for the table
    const explicitFeatureOrder = [
        '工作年齡人口比例', '家戶中位數所得', '第三產業比例', '醫療指數', '商業集中度指數' 
        // Ensure these strings exactly match the keys in DASHBOARD_DATA.districts[...].features
    ];


    sortedData.forEach((district) => {
        const row = tableBody.insertRow();
        const levelBadgeClass = district.level_en; // 'high', 'medium', 'low'
        
        row.className = `${levelBadgeClass}-row`; // e.g., high-row
        
        // Helper to get feature value or N/A, and format
        const getFeatureVal = (featureName, type) => {
            const val = district.features[featureName];
            if (typeof val !== 'number') return 'N/A';
            if (type === 'percent') return `${val.toFixed(1)}%`;
            if (type === 'currency') return val.toLocaleString('zh-TW', { maximumFractionDigits: 0 }) + ' 元'; // 保留元單位
            if (type === 'medical') return `<span title="醫療服務綜合指數：評估區域醫療資源密度與可及性，範圍0-1，數值越高表示醫療服務越完善">${val.toFixed(3)}</span>`; // 醫療指數tooltip
            if (type === 'hhi') return `<span title="商業集中度指數(HHI)：衡量區域商業活動分布均勻度，範圍0-10000，數值越高表示商業越集中">${val.toFixed(3)}</span>`; // HHI tooltip
            return val.toFixed(1); // Default
        };

        row.innerHTML = `
            <td>${district.rank}</td>
            <td>${district.name}</td>
            <td><span class="level-badge ${levelBadgeClass}">${district.level}</span></td>
            <td>${district.score.toFixed(1)}</td>
            <td>${getFeatureVal(explicitFeatureOrder[0], 'percent')}</td>
            <td>${getFeatureVal(explicitFeatureOrder[1], 'currency')}</td>
            <td>${getFeatureVal(explicitFeatureOrder[2], 'percent')}</td>
            <td>${getFeatureVal(explicitFeatureOrder[3], 'medical')}</td>
            <td>${getFeatureVal(explicitFeatureOrder[4], 'hhi')}</td> 
        `;
        
        // 添加hover tooltip
        row.addEventListener('mouseenter', function(e) {
            showRowTooltip(e, district);
        });
        
        row.addEventListener('mouseleave', function() {
            hideRowTooltip();
        });
    });
    console.log("詳細數據表格已填充。");
}

// 顯示行hover tooltip
function showRowTooltip(event, district) {
    hideRowTooltip(); // 先清除已存在的tooltip
    
    const tooltip = document.createElement('div');
    tooltip.id = 'row-tooltip';
    tooltip.className = 'row-tooltip';
    
    const rank = calculateOverallRank(district.name);
    const topFeature = getTopFeature(district);
    
    tooltip.innerHTML = `
        <div class="tooltip-header">
            <span class="tooltip-title">${district.name}</span>
        </div>
        <div class="tooltip-content">
            <ul class="tooltip-list">
                <li>• 綜合潛力: ${district.score.toFixed(1)}分 (${district.level})</li>
                <li>• 家戶所得: ${district.features['家戶中位數所得']?.toLocaleString('zh-TW', { maximumFractionDigits: 0 }) || 'N/A'} 元</li>
                <li>• 第三產業比重: ${district.features['第三產業比例']?.toFixed(1) || 'N/A'}%</li>
                <li>• 突出特徵: ${topFeature}</li>
            </ul>
        </div>
    `;
    
    document.body.appendChild(tooltip);
    
    // 位置計算
    const rect = event.target.closest('tr').getBoundingClientRect();
    tooltip.style.left = (rect.right + 10) + 'px';
    tooltip.style.top = (rect.top + window.scrollY - 10) + 'px';
    
    // 檢查是否超出視窗右邊界
    const tooltipRect = tooltip.getBoundingClientRect();
    if (tooltipRect.right > window.innerWidth - 10) {
        tooltip.style.left = (rect.left - tooltipRect.width - 10) + 'px';
    }
}

// 隱藏tooltip
function hideRowTooltip() {
    const tooltip = document.getElementById('row-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// 計算整體排名
function calculateOverallRank(districtName) {
    const district = DASHBOARD_DATA.districts.find(d => d.name === districtName);
    return district ? district.rank : 'N/A';
}

// 獲取最突出的特徵
function getTopFeature(district) {
    const features = district.features;
    const featureNames = Object.keys(features);
    
    // 找出數值最高的特徵（相對於平均值）
    let topFeature = '平均水準';
    let maxDiff = -Infinity;
    
    featureNames.forEach(featureName => {
        // 這裡可以加入與平均值的比較邏輯
        const value = features[featureName];
        if (typeof value === 'number' && value > maxDiff) {
            maxDiff = value;
            topFeature = featureName;
        }
    });
    
    return topFeature === '平均水準' ? '整體發展均衡' : topFeature;
}

function loadChartStatistics() {
    console.log('載入圖表統計...');
    
    // 設置tooltip說明
    const fStatElement = document.getElementById('fStatistic');
    const effectSizeElement = document.getElementById('effectSize');
    
    if (fStatElement) {
        fStatElement.setAttribute('title', 
            'F統計量指標說明：\n\n' +
            '• 計算公式：F = 組間方差 ÷ 組內方差\n\n' +
            '• 數值越大表示分級效果越好\n\n' +
            '• 代表不同潛力等級間的差異遠大於\n  各等級內部的差異\n\n' +
            '• 通常 F > 10 表示分級品質優秀'
        );
    }
    
    if (effectSizeElement) {
        effectSizeElement.setAttribute('title', 
            '效應大小 (η²) 指標說明：\n\n' +
            '• 計算公式：η² = 組間方差 ÷ (組間方差 + 組內方差)\n\n' +
            '• 範圍為 0-1，數值越接近 1 表示分級\n  解釋了越多的總變異\n\n' +
            '• η² > 0.8 被認為是大效應，\n  表示分級效果顯著'
        );
    }
    
    // 直接從 map_statistics.json 載入統計數據
    fetch('data/map_statistics.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(statistics => {
            console.log('統計數據載入成功:', statistics);
            
            // 更新F統計量
            if (fStatElement && statistics.f_statistic) {
                fStatElement.textContent = statistics.f_statistic.toFixed(3);
                console.log(`F統計量更新: ${statistics.f_statistic}`);
            }
            
            // 更新效應大小
            if (effectSizeElement && statistics.effect_size) {
                effectSizeElement.textContent = statistics.effect_size.toFixed(3);
                console.log(`效應大小更新: ${statistics.effect_size}`);
            }
        })
        .catch(error => {
            console.error('載入統計數據失敗:', error);
            
            // 顯示錯誤信息
            if (fStatElement) fStatElement.textContent = '載入失敗';
            if (effectSizeElement) effectSizeElement.textContent = '載入失敗';
        });
}

// 表格排序功能
let sortState = {};

function sortTable(columnIndex) {
    const table = document.getElementById('dataTable');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // 檢查當前排序狀態
    const currentOrder = sortState[columnIndex] || 'asc';
    const newOrder = currentOrder === 'asc' ? 'desc' : 'asc';
    sortState[columnIndex] = newOrder;
    
    // 更新所有排序圖標和標題狀態
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
        const icon = header.querySelector('i');
        if (index === columnIndex) {
            icon.className = newOrder === 'asc' ? 'fas fa-sort-up' : 'fas fa-sort-down';
            header.className = newOrder === 'asc' ? 'sorted-asc' : 'sorted-desc';
        } else {
            if (icon) icon.className = 'fas fa-sort';
            header.className = '';
        }
    });
    
    // 排序行
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        // 判斷是否為數字
        const aIsNumber = !isNaN(parseFloat(aValue.replace(/[,%]/g, '')));
        const bIsNumber = !isNaN(parseFloat(bValue.replace(/[,%]/g, '')));
        
        let result = 0;
        
        if (aIsNumber && bIsNumber) {
            // 數字排序
            const aNum = parseFloat(aValue.replace(/[,%]/g, ''));
            const bNum = parseFloat(bValue.replace(/[,%]/g, ''));
            result = aNum - bNum;
        } else {
            // 文字排序
            result = aValue.localeCompare(bValue, 'zh-TW');
        }
        
        return newOrder === 'asc' ? result : -result;
    });
    
    // 重新排列表格行
    rows.forEach(row => tbody.appendChild(row));
    
    // 更新排名欄位
    if (columnIndex !== 0) {
        rows.forEach((row, index) => {
            row.cells[0].textContent = index + 1;
        });
    }
}

// 錨點導航功能
function setupAnchorNavigation() {
    console.log('設置錨點導航...');
    
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('[id]');
    
    // 點擊導航項目
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const targetId = this.dataset.target;
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start',
                    inline: 'nearest'
                });
                
                // 更新active狀態
                navItems.forEach(nav => nav.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
    
    // 滾動時更新active狀態
    window.addEventListener('scroll', function() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (window.scrollY >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.target === current) {
                item.classList.add('active');
            }
        });
    });
}

// 響應式處理
window.addEventListener('resize', function() {
    if (scatterChart) {
        try { scatterChart.resize(); } catch (e) { console.warn("散點圖 resize 錯誤:", e); }
    }
    if (radarChart) {
        try { radarChart.resize(); } catch (e) { console.warn("雷達圖 resize 錯誤:", e); }
    }
});

// 導出函數供外部使用 (如果需要)
// window.JenksDashboard = {
//     updateRadarChart,
//     clearRadarChart,
//     sortTable
// }; 