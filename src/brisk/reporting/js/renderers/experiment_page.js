class ExperimentPageRenderer {
    constructor(experimentData) {
        this.experimentData = experimentData;
        this.selectedAlgorithmIndex = 0;
        this.selectedTableIndex = 0;
        this.selectedPlotIndex = 0;
    }

    render() {
        const template = document.getElementById('experiment-template').content.cloneNode(true);
        this.renderExperimentSummary(template);
        this.renderExperimentTables(template);
        this.renderExperimentPlots(template);
        return template;
    }

    renderExperimentSummary(template) {
        this.renderAlgorithmSelector(template);
        this.updateAlgorithmTitle(template);
        this.renderTunedHyperparams(template);
        this.renderHyperparamGrid(template);
        this.renderDataSplits(template);
    }

    renderExperimentTables(template) {
        const container = template.querySelector('.experiment-tables');
        
        if (!this.experimentData.tables || this.experimentData.tables.length === 0) {
            container.innerHTML = '<p>No tables available</p>';
            return;
        }

        const tablesList = container.querySelector('.tables-list');
        tablesList.innerHTML = '';

        const tablesContent = container.querySelector('.tables-content');

        this.experimentData.tables.forEach((tableData, index) => {
            const tableName = document.createElement('div');
            tableName.textContent = tableData.name;
            tableName.className = `table-name ${index === this.selectedTableIndex ? 'selected' : ''}`;
            tableName.dataset.tableIndex = index;
            
            tablesList.appendChild(tableName);
        });
        
        this.renderSelectedTable(tablesContent);
    }

    renderSelectedTable(contentContainer) {
        contentContainer.innerHTML = '';
        
        const selectedTable = this.experimentData.tables[this.selectedTableIndex];
        if (selectedTable) {
            const tableRenderer = new TableRenderer(selectedTable);
            const tableElement = tableRenderer.render();
            contentContainer.appendChild(tableElement);
        }
    }

    renderExperimentPlots(template) {
        const container = template.querySelector('.experiment-plots');
        
        if (!this.experimentData.plots || this.experimentData.plots.length === 0) {
            container.innerHTML = '<p>No plots available</p>';
            return;
        }

        const plotsList = container.querySelector('.plots-list');
        plotsList.innerHTML = '';

        const plotsContent = container.querySelector('.plots-content');
        
        this.experimentData.plots.forEach((plotData, index) => {
            const plotName = document.createElement('div');
            plotName.textContent = plotData.name;
            plotName.className = `plot-name ${index === this.selectedPlotIndex ? 'selected' : ''}`;
            plotName.dataset.plotIndex = index;
            
            plotsList.appendChild(plotName);
        });
        
        this.renderSelectedPlot(plotsContent);
    }

    renderSelectedPlot(contentContainer) {
        contentContainer.innerHTML = '';
        
        const selectedPlot = this.experimentData.plots[this.selectedPlotIndex];
        if (selectedPlot) {
            const plotRenderer = new PlotRenderer(selectedPlot);
            const plotElement = plotRenderer.render();
            contentContainer.appendChild(plotElement);
        }
    }

    renderAlgorithmSelector(template) {
        const selectorContainer = template.querySelector('.algorithm-selector-container');

        // Only show selector if there are multiple algorithms
        if (this.experimentData.algorithm.length <= 1) {
            selectorContainer.style.display = 'none';
            return;
        }

        selectorContainer.style.display = 'block';        
        const algorithmNav = selectorContainer.querySelector('.algorithm-nav');        
        algorithmNav.innerHTML = '';
        
        this.experimentData.algorithm.forEach((algorithm, index) => {
            const algorithmText = document.createElement('span');
            algorithmText.textContent = algorithm;
            algorithmText.className = `algorithm-text ${index === this.selectedAlgorithmIndex ? 'selected' : ''}`;
            algorithmText.dataset.algorithmIndex = index;

            algorithmNav.appendChild(algorithmText);
        });
    }

    updateAlgorithmTitle(template) {
        const titleElement = template.querySelector('.experiment-title');
        const selectedAlgorithm = this.experimentData.algorithm[this.selectedAlgorithmIndex];
        titleElement.textContent = `Algorithm: ${selectedAlgorithm}`;
    }

    renderTunedHyperparams(template) {
        const container = template.querySelector('.tuned-hyperparams');
        const message = container.querySelector('.tuned-hyperparams-message');
        const table = container.querySelector('.tuned-hyperparams-table');
        const tbody = container.querySelector('.tuned-hyperparams-body');
        
        if (!this.experimentData.tuned_params || 
            Object.keys(this.experimentData.tuned_params).length === 0) {
            message.style.display = 'block';
            table.style.display = 'none';
            return;
        }

        message.style.display = 'none';
        table.style.display = 'table';
        tbody.innerHTML = '';

        Object.entries(this.experimentData.tuned_params).forEach(([param, value]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${param}</td>
                <td>${value}</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderHyperparamGrid(template) {
        const container = template.querySelector('.hyperparam-grid');
        const message = container.querySelector('.hyperparam-grid-message');
        const table = container.querySelector('.hyperparam-grid-table');
        const tbody = container.querySelector('.hyperparam-grid-body');
        
        if (!this.experimentData.hyperparam_grid || 
            Object.keys(this.experimentData.hyperparam_grid).length === 0) {
            message.style.display = 'block';
            table.style.display = 'none';
            return;
        }

        message.style.display = 'none';
        table.style.display = 'table';
        tbody.innerHTML = '';

        Object.entries(this.experimentData.hyperparam_grid).forEach(([param, value]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${param}</td>
                <td>${value}</td>
            `;
            tbody.appendChild(row);
        });
    }

    renderDataSplits(template) {
        const container = template.querySelector('.data-splits');
        const splitsNav = container.querySelector('.splits-nav');        
        splitsNav.innerHTML = '';
        
        const numSplits = window.app.reportData.datasets[this.experimentData.dataset].splits.length;
        
        for (let i = 0; i < numSplits; i++) {
            const splitText = document.createElement('span');
            splitText.textContent = `Split ${i}`;
            splitText.className = `split-text ${i === 0 ? 'selected' : ''}`;
            splitText.dataset.split = i;
            
            splitsNav.appendChild(splitText);
        }
    }
}