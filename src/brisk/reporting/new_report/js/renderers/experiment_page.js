class ExperimentPageRenderer {
    constructor(experimentData) {
        this.experimentData = experimentData;
        this.selectedAlgorithmIndex = 0; // Track selected algorithm
        this.selectedTableIndex = 0; // Track selected table
        this.selectedPlotIndex = 0; // Track selected plot
    }

    render() {
        const template = document.getElementById('experiment-template').content.cloneNode(true);
        this.renderExperimentSummary(template);
        this.renderExperimentTables(template);
        this.renderExperimentPlots(template);
        return template;
    }

    renderExperimentSummary(template) {
        // Render algorithm selector if multiple algorithms
        this.renderAlgorithmSelector(template);
        
        // Set the algorithm title (will show selected algorithm)
        this.updateAlgorithmTitle(template);

        // Render tuned hyperparameters section
        this.renderTunedHyperparams(template);
        
        // Render hyperparameter grid section
        this.renderHyperparamGrid(template);
        
        // Render data splits section
        this.renderDataSplits(template);
    }

    renderExperimentTables(template) {
        const container = template.querySelector('.experiment-tables');
        
        if (!this.experimentData.tables || this.experimentData.tables.length === 0) {
            container.innerHTML = '<p>No tables available</p>';
            return;
        }

        // Create the layout structure
        const tablesLayout = document.createElement('div');
        tablesLayout.className = 'tables-layout';
        
        // Left side - table navigation
        const tablesNav = document.createElement('div');
        tablesNav.className = 'tables-nav';
        
        const tablesTitle = document.createElement('h3');
        tablesTitle.textContent = 'Tables';
        tablesTitle.className = 'section-title';
        tablesNav.appendChild(tablesTitle);
        
        const tablesList = document.createElement('div');
        tablesList.className = 'tables-list';
        
        this.experimentData.tables.forEach((tableData, index) => {
            const tableName = document.createElement('div');
            tableName.textContent = tableData.name;
            tableName.className = `table-name ${index === this.selectedTableIndex ? 'selected' : ''}`;
            tableName.dataset.tableIndex = index;
            
            tableName.addEventListener('click', () => {
                this.selectedTableIndex = index;
                
                // Update active state
                tablesList.querySelectorAll('.table-name').forEach(name => {
                    name.classList.remove('selected');
                });
                tableName.classList.add('selected');
                
                // Render the selected table
                this.renderSelectedTable(tablesContent);
            });
            
            tablesList.appendChild(tableName);
        });
        
        tablesNav.appendChild(tablesList);
        
        // Right side - table content
        const tablesContent = document.createElement('div');
        tablesContent.className = 'tables-content';
        
        // Render initial table
        this.renderSelectedTable(tablesContent);
        
        tablesLayout.appendChild(tablesNav);
        tablesLayout.appendChild(tablesContent);
        container.appendChild(tablesLayout);
    }

    renderSelectedTable(contentContainer) {
        contentContainer.innerHTML = ''; // Clear previous content
        
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

        // Create the layout structure
        const plotsLayout = document.createElement('div');
        plotsLayout.className = 'plots-layout';
        
        // Left side - plot navigation
        const plotsNav = document.createElement('div');
        plotsNav.className = 'plots-nav';
        
        const plotsTitle = document.createElement('h3');
        plotsTitle.textContent = 'Plots';
        plotsTitle.className = 'section-title';
        plotsNav.appendChild(plotsTitle);
        
        const plotsList = document.createElement('div');
        plotsList.className = 'plots-list';
        
        this.experimentData.plots.forEach((plotData, index) => {
            const plotName = document.createElement('div');
            plotName.textContent = plotData.name;
            plotName.className = `plot-name ${index === this.selectedPlotIndex ? 'selected' : ''}`;
            plotName.dataset.plotIndex = index;
            
            plotName.addEventListener('click', () => {
                this.selectedPlotIndex = index;
                
                // Update active state
                plotsList.querySelectorAll('.plot-name').forEach(name => {
                    name.classList.remove('selected');
                });
                plotName.classList.add('selected');
                
                // Render the selected plot
                this.renderSelectedPlot(plotsContent);
            });
            
            plotsList.appendChild(plotName);
        });
        
        plotsNav.appendChild(plotsList);
        
        // Right side - plot content
        const plotsContent = document.createElement('div');
        plotsContent.className = 'plots-content';
        
        // Render initial plot
        this.renderSelectedPlot(plotsContent);
        
        plotsLayout.appendChild(plotsNav);
        plotsLayout.appendChild(plotsContent);
        container.appendChild(plotsLayout);
    }

    renderSelectedPlot(contentContainer) {
        contentContainer.innerHTML = ''; // Clear previous content
        
        const selectedPlot = this.experimentData.plots[this.selectedPlotIndex];
        if (selectedPlot) {
            const plotRenderer = new PlotRenderer(selectedPlot);
            const plotElement = plotRenderer.render();
            contentContainer.appendChild(plotElement);
        }
    }

    renderAlgorithmSelector(template) {
        // Only show selector if there are multiple algorithms
        if (this.experimentData.algorithm.length <= 1) {
            return;
        }

        // Find the first hr and insert algorithm selector before it
        const firstHr = template.querySelector('hr');
        
        // Create algorithm selector container
        const selectorContainer = document.createElement('div');
        selectorContainer.className = 'algorithm-selector-container';
        
        // Create algorithm navigation using same styling as splits
        const algorithmNav = document.createElement('div');
        algorithmNav.className = 'algorithm-nav';
        
        // Create algorithm text elements
        this.experimentData.algorithm.forEach((algorithm, index) => {
            const algorithmText = document.createElement('span');
            algorithmText.textContent = algorithm;
            algorithmText.className = `algorithm-text ${index === this.selectedAlgorithmIndex ? 'selected' : ''}`;
            algorithmText.dataset.algorithmIndex = index;
            
            // Add click handler for algorithm selection
            algorithmText.addEventListener('click', () => {
                // Update selected algorithm index
                this.selectedAlgorithmIndex = index;
                
                // Update active state
                algorithmNav.querySelectorAll('.algorithm-text').forEach(text => {
                    text.classList.remove('selected');
                });
                algorithmText.classList.add('selected');
                
                // Update the algorithm title
                this.updateAlgorithmTitle(template);                
            });
            
            algorithmNav.appendChild(algorithmText);
        });
        
        selectorContainer.appendChild(algorithmNav);
        
        // Insert before the first hr
        firstHr.parentNode.insertBefore(selectorContainer, firstHr);
    }

    updateAlgorithmTitle(template) {
        const titleElement = template.querySelector('.experiment-title');
        const selectedAlgorithm = this.experimentData.algorithm[this.selectedAlgorithmIndex];
        titleElement.textContent = `Algorithm: ${selectedAlgorithm}`;
    }

    // Reusable table creation method using split-table styling
    createParameterTable(title, data, valueColumnTitle = 'Value') {
        const container = document.createElement('div');
        container.className = 'parameter-section';
        
        // Create section title
        const titleElement = document.createElement('h3');
        titleElement.textContent = title;
        titleElement.className = 'parameter-title';
        
        // Create table using split-table styling
        const table = document.createElement('table');
        table.className = 'split-table parameter-table';
        
        // Create header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Parameter</th>
                <th>${valueColumnTitle}</th>
            </tr>
        `;
        table.appendChild(thead);
        
        // Create body
        const tbody = document.createElement('tbody');
        Object.entries(data).forEach(([param, value]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${param}</td>
                <td>${value}</td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        
        container.appendChild(titleElement);
        container.appendChild(table);
        return container;
    }

    renderTunedHyperparams(template) {
        const container = template.querySelector('.tuned-hyperparams');
        const table = this.createParameterTable(
            'Tuned Hyperparameters', 
            this.experimentData.tuned_params
        );
        container.appendChild(table);
    }

    renderHyperparamGrid(template) {
        const container = template.querySelector('.hyperparam-grid');
        const table = this.createParameterTable(
            'Hyperparameter Grid',
            this.experimentData.hyperparam_grid,
            'Values'
        );
        container.appendChild(table);
    }

    renderDataSplits(template) {
        const container = template.querySelector('.data-splits');
        
        // Get the actual number of splits from the dataset
        const numSplits = window.app.reportData.datasets[this.experimentData.dataset].splits.length;
        
        // Create splits navigation using dataset-name styling
        const splitsNav = document.createElement('div');
        splitsNav.className = 'splits-nav';
        
        // Create split text elements
        for (let i = 0; i < numSplits; i++) {
            const splitText = document.createElement('span');
            splitText.textContent = `Split ${i}`;
            splitText.className = `split-text ${i === 0 ? 'selected' : ''}`;
            splitText.dataset.split = i;
            
            // Add click handler for split navigation
            splitText.addEventListener('click', () => {
                // Update active state
                splitsNav.querySelectorAll('.split-text').forEach(text => {
                    text.classList.remove('selected');
                });
                splitText.classList.add('selected');
            });
            
            splitsNav.appendChild(splitText);
        }
        
        container.appendChild(splitsNav);
    }
}