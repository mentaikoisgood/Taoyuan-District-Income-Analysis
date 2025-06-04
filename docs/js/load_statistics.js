// 載入統計數據並更新界面
// 解決 "F統計量: 加載中..." 和 "效應大小: 加載中..." 的問題

document.addEventListener('DOMContentLoaded', function() {
    loadStatistics();
});

async function loadStatistics() {
    try {
        console.log('🔢 載入統計數據...');
        
        // 載入map_statistics.json
        const response = await fetch('data/map_statistics.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const statistics = await response.json();
        console.log('✅ 統計數據載入成功:', statistics);
        
        // 更新F統計量
        const fStatElement = document.getElementById('fStatistic');
        if (fStatElement) {
            fStatElement.textContent = statistics.f_statistic;
            console.log(`✅ F統計量更新: ${statistics.f_statistic}`);
        }
        
        // 更新效應大小
        const effectSizeElement = document.getElementById('effectSize');
        if (effectSizeElement) {
            effectSizeElement.textContent = statistics.effect_size;
            console.log(`✅ 效應大小更新: ${statistics.effect_size}`);
        }
        
        // 更新洞察中的平均分數
        updateInsightCards(statistics);
        
    } catch (error) {
        console.error('❌ 載入統計數據失敗:', error);
        
        // 顯示錯誤信息
        const fStatElement = document.getElementById('fStatistic');
        const effectSizeElement = document.getElementById('effectSize');
        
        if (fStatElement) fStatElement.textContent = '載入失敗';
        if (effectSizeElement) effectSizeElement.textContent = '載入失敗';
    }
}

function updateInsightCards(statistics) {
    try {
        // 更新高潛力區域平均分數
        const highAvgElement = document.getElementById('highAvgScore');
        if (highAvgElement && statistics.level_statistics?.高潛力) {
            highAvgElement.textContent = statistics.level_statistics.高潛力.avg_score;
        }
        
        // 更新中潛力區域平均分數
        const mediumAvgElement = document.getElementById('mediumAvgScore');
        if (mediumAvgElement && statistics.level_statistics?.中潛力) {
            mediumAvgElement.textContent = statistics.level_statistics.中潛力.avg_score;
        }
        
        // 更新低潛力區域平均分數
        const lowAvgElement = document.getElementById('lowAvgScore');
        if (lowAvgElement && statistics.level_statistics?.低潛力) {
            lowAvgElement.textContent = statistics.level_statistics.低潛力.avg_score;
        }
        
        console.log('✅ 洞察卡片數據更新完成');
        
    } catch (error) {
        console.error('❌ 更新洞察卡片失敗:', error);
    }
}

// 導出函數供其他模塊使用
window.StatisticsLoader = {
    loadStatistics,
    updateInsightCards
}; 