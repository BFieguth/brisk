/**
 * Implements the logic to make page interactive.
 * @class
 * @property {Object} reportData - JSON object with the data to render.
 * @property {string} currentPage - The current page type. Default is 'home'.
 * @property {Object} pendingTableUpdate - The pending table update with the table data and timestamp. Default is null.
 * @property {boolean} isProcessingAnimation - Whether the animation is processing. Default is false.
 * @property {number} currentExperimentGroupCard - The current experiment group card index. Default is 0.
 * @property {Dataset} selectedDataset - The selected dataset instance.
 * @property {number} selectedSplit - The selected split index on home page. Default is 0.
 * @property {string} currentExperimentId - The current experiment ID. Default is null.
 * @property {number} selectedAlgorithmIndex - The selected algorithm index. Default is 0.
 * @property {number} selectedSplitIndex - The selected data split index on experiment page. Default is 0.
 * @property {number} selectedTableIndex - The selected table index on experiment page. Default is 0.
 * @property {number} selectedPlotIndex - The selected plot index on experiment page. Default is 0.
 * @property {Array} currentDatasetSplits - The splits of the selected dataset. Default is an empty array.
 * @property {string} currentDatasetId - The current dataset ID. Default is null.
 * @property {number} selectedDatasetSplitIndex - The selected data split index on dataset page. Default is 0.
 * @property {number} selectedFeatureIndex - The selected feature index on dataset page. Default is 0.
 */
class App {
    /**
     * @param {Object} reportData - JSON object with the data to render.
     * @constructor
     */
    constructor(reportData) {
        this.reportData = reportData;
        this.currentPage = 'home';
        this.pendingTableUpdate = null;
        this.isProcessingAnimation = false;

        //  Home page state
        this.currentExperimentGroupCard = 0;
        this.selectedDataset = null;
        this.selectedSplit = 0;
        this.initalizeHomeSelections();

        // Experiment page state
        this.currentExperimentId = null;
        this.selectedAlgorithmIndex = 0;
        this.selectedSplitIndex = 0;
        this.selectedTableIndex = 0;
        this.selectedPlotIndex = 0;
        this.currentDatasetSplits = [];

        // Dataset page state
        this.currentDatasetId = null;
        this.selectedDatasetSplitIndex = 0;
        this.selectedFeatureIndex = 0;
    }

    /**
     * Initialize the app, setup theme, navigation, and show the home page.
     */
    init() {
        this.initializeTheme();
        this.showPage('home');
        this.setupNavigation();
    }

    /**
     * Set this.selectedDataset to the first dataset of the first experiment group.
     */
    initalizeHomeSelections() {
        const experimentGroups = this.reportData.experiment_groups;
        if (experimentGroups.length > 0) {
            const firstGroup = experimentGroups[0];
            // const datasetID = firstGroup.datasets[0];
            const datasetID = firstGroup.name + "_" + firstGroup.datasets[0];
            this.selectedDataset = this.reportData.datasets[datasetID];
        }
    }

    /**
     * Toggle the theme between light and dark.
     */
    toggleTheme() {
        const root = document.documentElement;
        const themeText = document.querySelector('.theme-text');
        const darkIcon = document.querySelector('.dark-mode-icon');
        const lightIcon = document.querySelector('.light-mode-icon');
        
        if (root.classList.contains('light-theme')) {
            // Switch to dark theme
            root.classList.remove('light-theme');
            themeText.textContent = 'Light Mode';
            lightIcon.style.display = 'block';
            darkIcon.style.display = 'none';
            localStorage.setItem('theme', 'dark');
        } else {
            // Switch to light theme
            root.classList.add('light-theme');
            themeText.textContent = 'Dark Mode';
            lightIcon.style.display = 'none';
            darkIcon.style.display = 'block';
            localStorage.setItem('theme', 'light');
        }
    }

    /**
     * Initialize the theme based on the saved theme or the system preference.
     */
    initializeTheme() {
        const savedTheme = localStorage.getItem('theme');
        const root = document.documentElement;
        const themeText = document.querySelector('.theme-text');
        const darkIcon = document.querySelector('.dark-mode-icon');
        const lightIcon = document.querySelector('.light-mode-icon');
        
        let themeToApply;

        if (savedTheme) {
            themeToApply = savedTheme;
        } else {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            themeToApply = prefersDark ? 'dark' : 'light';
            localStorage.setItem('theme', themeToApply);
        }

        if (savedTheme === 'light') {
            root.classList.add('light-theme');
            themeText.textContent = 'Dark Mode';
            lightIcon.style.display = 'none';
            darkIcon.style.display = 'block';
        } else {
            root.classList.remove('light-theme');
            themeText.textContent = 'Light Mode';
            lightIcon.style.display = 'block';
            darkIcon.style.display = 'none';
        }
    }

    /**
     * Display the page based on the page type and data.
     * @param {string} pageType - The type of page to render.
     * @param {Object} pageData - The data to render.
     */
    showPage(pageType, pageData = null) {
        const mainContent = document.getElementById('main-content');
        this.currentPage = pageType;

        try {
            let content = this.renderPage(pageType, pageData);
            mainContent.innerHTML = content;

            // Handle experiment navigation
            if (pageType === 'experiment' && pageData) {
                this.initializeExperimentPage(pageData);
                this.showExperimentNavigation(pageData);
            } else {
                this.hideExperimentNavigation();
            }

            if (pageType === 'home') {
                // Ensure DOM is ready first
                setTimeout(() => this.initializeCarousel(), 50);
            }

            if (pageType === 'dataset') {
                // Initialize dataset page interactivity
                this.initializeDatasetPage();
            }
        } catch (e) {
            console.error('Error rendering page:', e);
            mainContent.innerHTML = '<div>Error loading page</div>';
        }
    }

    /**
     * Render the page based on the page type. Pass data to the appropriate renderer.
     * @param {string} pageType - The type of page to render.
     * @param {Object} pageData - The data to render.
     * @returns {string} - The HTML content of the page.
     */
    renderPage(pageType, pageData = null) {
        switch(pageType) {
            case 'home':
                return this.renderHomePage();
            case 'experiment':
                return this.renderExperimentPage(pageData);
            case 'dataset':
                return this.renderDatasetPage();
            default:
                return '<div>Renderer not found</div>';
        }
    }

    /**
     * Render the home page using HomeRenderer.
     * @returns {string} - The HTML content of the home page.
     */
    renderHomePage() {
        const selectedTable = this.getCurrentSelectedTableData();
        const renderer = new HomeRenderer(
            this.reportData.experiment_groups, selectedTable
        );
        const renderedElement = renderer.render();
        const tempDiv = document.createElement('div');
        tempDiv.appendChild(renderedElement);
        return tempDiv.innerHTML;
    }

    /**
     * Render the experiment page using ExperimentPageRenderer.
     * @param {string} experimentData - The experiment ID.
     * @returns {string} - The HTML content of the experiment page.
     */
    renderExperimentPage(experimentData) {
        const experimentInstance = this.reportData.experiments[experimentData]
        const renderer = new ExperimentPageRenderer(experimentInstance);
        const renderedElement = renderer.render();
        const tempDiv = document.createElement('div');
        tempDiv.appendChild(renderedElement);
        return tempDiv.innerHTML;
    }

    /**
     * Render the dataset page using DatasetPageRenderer.
     * @returns {string} - The HTML content of the dataset page.
     */
    renderDatasetPage() {
        const renderer = new DatasetPageRenderer(this.selectedDataset);
        const renderedElement = renderer.render();
        const tempDiv = document.createElement('div');
        tempDiv.appendChild(renderedElement);
        return tempDiv.innerHTML;

    }

    /**
     * Setup event listener to detect clicks on navigation links.
     */
    setupNavigation() {
        const self = this;
        document.addEventListener('click', function(event) {
            if (event.target.matches('[page-type]')) {
                event.preventDefault();
                const pageType = event.target.getAttribute('page-type');
                const pageData = event.target.getAttribute('page-data');
                self.showPage(pageType, pageData);
            }
        });
    }

    /**
     * Select a dataset on the home page. Changes the rendered card and table.
     * @param {Element} clickedElement - The element that was clicked.
     * @param {string} datasetName - The name of the dataset.
     * @param {number} cardIndex - The index of the card.
     */
    selectDataset(clickedElement, datasetName, cardIndex) {
        // Select Dataset (Home ExperimentGroup Cards)
        const card = clickedElement.closest('.experiment-group-card');
        const allDatasetItems = card.querySelectorAll('.dataset-name');
        const groupName = this.reportData.experiment_groups[cardIndex];
        const datasetID = `${groupName.name}_${datasetName}`;
        const datasetInstance = this.reportData.datasets[datasetID];
        allDatasetItems.forEach(item => item.classList.remove('selected'));

        // Add selected class to clicked item
        clickedElement.classList.add('selected');

        this.selectedDataset = datasetInstance;
        this.selectedSplit = 0;

        this.updateDataSplitsTable(cardIndex, datasetInstance.ID);
        this.updateHomeTables();
        
        // Update experiment list for the selected dataset
        this.updateExperimentList(card, cardIndex, datasetInstance.ID);

        // Prevent default link behavior
        return false;
    }

    /**
     * Update the experiment list on the home page card.
     * @param {Element} card - The card element.
     * @param {number} cardIndex - The index of the card.
     * @param {string} datasetID - The ID of the dataset.
     */
    updateExperimentList(card, cardIndex, datasetID) {
        const experimentList = card.querySelector('.experiment-list');
        if (!experimentList) return;

        // Clear existing experiments
        experimentList.innerHTML = '';

        const experimentGroup = this.reportData.experiment_groups[cardIndex];
        if (!experimentGroup) return;

        // Get experiments for the selected dataset
        const datasetExperiments = experimentGroup.experiments.filter(expId => {
            const exp = this.reportData.experiments[expId];
            return exp && exp.dataset === datasetID;
        });

        // Group by algorithm name
        const baseExperiments = new Map(); // Using Map to store algorithm name -> full ID mapping
        datasetExperiments.forEach(experimentID => {
            const exp = this.reportData.experiments[experimentID];
            const baseExpName = exp.algorithm.join('_');
            
            if (!baseExperiments.has(baseExpName)) {
                baseExperiments.set(baseExpName, experimentID);
            }
        });

        // Create links for each unique base experiment
        Array.from(baseExperiments.entries()).sort().forEach(([baseExpName, fullExperimentID]) => {
            const listItem = document.createElement('li');

            const experimentAnchor = document.createElement('a');
            experimentAnchor.href = '#';
            experimentAnchor.setAttribute('page-type', 'experiment');
            experimentAnchor.setAttribute('page-data', fullExperimentID);
            experimentAnchor.className = 'experiment-link';
            
            // Use the algorithm name directly
            const exp = this.reportData.experiments[fullExperimentID];
            const cleanName = exp.algorithm.join('_').replace(/_/g, ' ');
            experimentAnchor.textContent = cleanName;
            
            listItem.appendChild(experimentAnchor);
            experimentList.appendChild(listItem);
        });
    }

    /**
     * Get the current experiment group object.
     * @returns {Object} - The current experiment group.
     */
    getCurrentExperimentGroup() {
        const experimentGroups = this.reportData.experiment_groups;
        if (experimentGroups.length > this.currentExperimentGroupCard) {
            return experimentGroups[this.currentExperimentGroupCard];
        }
        return null;
    }

    /**
     * Get the current selected table data.
     * @returns {Object} - The current selected table data.
     */
    getCurrentSelectedTableData() {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup || !this.selectDataset) {
            return null;
        }
        const tableKey = `${this.selectedDataset.ID}_split_${this.selectedSplit}`;
        return experimentGroup.test_scores[tableKey] || null;
    }

    /**
     * Update the data splits table on the home page card.
     * @param {number} cardIndex - The index of the card.
     * @param {string} datasetID - The ID of the dataset.
     */
    updateDataSplitsTable(cardIndex, datasetID) {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup) return;

        const splitTable = document.getElementById(`split-table-${cardIndex}`);

        if (!splitTable || !experimentGroup.data_split_scores[datasetID]) return;

        const splitData = experimentGroup.data_split_scores[datasetID];
        
        // Update table header with correct metric
        const thead = splitTable.querySelector('thead');
        let metricName = "Score"; // Default fallback
        if (splitData && splitData.length > 0) {
            const firstSplit = splitData[0];
            if (firstSplit.length > 3) {
                metricName = firstSplit[3];
            }
        }
        
        thead.innerHTML = `
            <tr>
                <th>Split</th>
                <th>Best Algorithm</th>
                <th>${metricName}</th>
            </tr>
        `;

        const tbody = splitTable.querySelector('tbody');
        tbody.innerHTML = '';

        splitData.forEach((split, index) => {
            const [splitName, algorithm, score] = split;
            const row = document.createElement('tr');
            row.className = 'split-row';
            row.setAttribute('data-split-index', index);
            if (index === this.selectedSplit) {
                row.classList.add('selected');
            }
            row.setAttribute('onclick', `window.app.selectSplit(this, ${index})`);

            row.innerHTML = `
                <td class="split-name">${splitName}</td>
                <td class="algorithm">${algorithm}</td>
                <td class="score">${score}</td>
            `;

            tbody.appendChild(row);
        });
    }

    /**
     * Select a split on the home page card.
     * @param {Element} clickedRow - The row that was clicked.
     * @param {number} splitIndex - The index of the split.
     */
    selectSplit(clickedRow, splitIndex) {
        const table = clickedRow.closest('table');
        const allRows = table.querySelectorAll('.split-row');
        allRows.forEach(row => row.classList.remove('selected'));

        clickedRow.classList.add('selected');
        this.selectedSplit = splitIndex;
        this.updateHomeTables();
    }

    /**
     * Add a pending table update to the queue.
     */
    updateHomeTables() {
        const selectedTable = this.getCurrentSelectedTableData();
        
        this.pendingTableUpdate = {
            tableData: selectedTable,
            timestamp: Date.now()
        };

        if (!this.isProcessingAnimation) {
            this.processNextTableUpdate();
        }
    }

    /**
     * Process the next table update in the queue if an animation is not already processing.
     */
    async processNextTableUpdate() {
        if (this.isProcessingAnimation) return;
        
        this.isProcessingAnimation = true;

        // Keep processing until no more pending updates
        while (this.pendingTableUpdate) {
            const currentRequest = this.pendingTableUpdate;
            this.pendingTableUpdate = null; // Clear pending before processing
            await this.executeTableUpdate(currentRequest.tableData);
        }

        this.isProcessingAnimation = false;
    }

    /**
     * Render the new table and animate the transition.
     * @param {Object} selectedTable - The selected table data.
     * @returns {Promise} - A promise that resolves when the table update is complete.
     */
    executeTableUpdate(selectedTable) {
        return new Promise((resolve) => {
            const tablesContainer = document.querySelector('.tables-container');

            if (!tablesContainer) {
                resolve();
                return;
            }

            if (!selectedTable) {
                this.animateTableTransition(tablesContainer, null, resolve);
                return;
            }

            try {
                const tableRenderer = new TableRenderer(selectedTable);
                const newTableElement = tableRenderer.render();
                
                if (!newTableElement || !newTableElement.firstElementChild) {
                    console.error('TableRenderer failed to create valid element');
                    resolve();
                    return;
                }
                this.animateTableTransition(tablesContainer, newTableElement.firstElementChild, resolve);
            } catch (error) {
                console.error('Error creating table:', error);
                resolve();
            }
        });
    }

    /**
     * Animate the table transition.
     * @param {Element} container - The container element.
     * @param {Element} newTableElement - The new table element.
     * @param {Function} onComplete - The callback function to call when the transition is complete.
     */
    animateTableTransition(container, newTableElement, onComplete) {
        if (this.pendingAnimationTimeout) {
            clearTimeout(this.pendingAnimationTimeout);
            this.pendingAnimationTimeout = null;
        }
        this.forceCleanupAnimation(container);

        const currentContent = container.firstElementChild;        
        if (!currentContent && !newTableElement) {
            onComplete();
            return;
        }
        
        if (!currentContent && newTableElement) {
            container.setAttribute('data-transition', 'fade-in');
            container.appendChild(newTableElement);
            
            this.pendingAnimationTimeout = setTimeout(() => {
                container.setAttribute('data-transition', 'idle');
                this.pendingAnimationTimeout = null;
                onComplete();
            }, 600);
            return;
        }
        
        if (!newTableElement) {
            container.setAttribute('data-transition', 'fade-out');
            
            this.pendingAnimationTimeout = setTimeout(() => {
                container.innerHTML = '';
                container.style.height = '';
                container.setAttribute('data-transition', 'idle');
                this.pendingAnimationTimeout = null;
                onComplete();
            }, 500);
            return;
        }        
        this.performSmoothTableSwap(container, newTableElement, onComplete);
    }

    /**
     * Force the cleanup of the animation.
     * @param {Element} container - The container element.
     */
    forceCleanupAnimation(container) {
        const oldContents = container.querySelectorAll('.old-content');
        oldContents.forEach(content => {
            if (content.parentNode === container) {
                container.removeChild(content);
            }
        });

        const newContents = container.querySelectorAll('.new-content');
        newContents.forEach(content => {
            content.classList.remove('new-content', 'fade-in');
            content.style.position = '';
            content.style.top = '';
            content.style.left = '';
            content.style.right = '';
        });

        container.style.height = '';
        container.setAttribute('data-transition', 'idle');
    }

    /**
     * Perform smooth table swap animation.
     * @param {Element} container - The container element.
     * @param {Element} newTableElement - The new table element.
     * @param {Function} onComplete - The callback function to call when the transition is complete.
     */
    performSmoothTableSwap(container, newTableElement, onComplete) {
        const currentContent = container.firstElementChild;
        if (!newTableElement || !newTableElement.classList) {
            console.error('Invalid new table element');
            onComplete();
            return;
        }
        
        if (!currentContent) {
            container.setAttribute('data-transition', 'fade-in');
            container.appendChild(newTableElement);
            
            this.pendingAnimationTimeout = setTimeout(() => {
                container.setAttribute('data-transition', 'idle');
                this.pendingAnimationTimeout = null;
                onComplete();
            }, 600);
            return;
        }
        
        const currentHeight = container.offsetHeight;        
        const tempContainer = document.createElement('div');
        tempContainer.style.cssText = `
            position: absolute;
            visibility: hidden;
            width: ${container.offsetWidth}px;
            top: -9999px;
            left: -9999px;
        `;
        tempContainer.appendChild(newTableElement.cloneNode(true));
        document.body.appendChild(tempContainer);
        const newHeight = tempContainer.offsetHeight;
        document.body.removeChild(tempContainer);
        
        container.setAttribute('data-transition', 'crossfade');
        container.style.height = currentHeight + 'px';        
        if (currentContent.classList) {
            currentContent.classList.add('old-content');
        }
        newTableElement.classList.add('new-content');        
        container.appendChild(newTableElement);
        
        requestAnimationFrame(() => {
            container.style.height = newHeight + 'px';            
            if (currentContent.classList) {
                currentContent.classList.add('fade-out');
            }
            newTableElement.classList.add('fade-in');
            
            this.pendingAnimationTimeout = setTimeout(() => {
                if (currentContent && currentContent.parentNode === container) {
                    container.removeChild(currentContent);
                }
                
                if (newTableElement.classList) {
                    newTableElement.classList.remove('new-content', 'fade-in');
                }
                newTableElement.style.position = '';
                newTableElement.style.top = '';
                newTableElement.style.left = '';
                newTableElement.style.right = '';
                
                container.style.height = '';
                container.setAttribute('data-transition', 'idle');
                this.pendingAnimationTimeout = null;
                onComplete();
            }, 800);
        });
    }

    /**
     * Update the carousel to show the selected experiment group card.
     */
    updateCarousel() {
        const cards = document.querySelectorAll(".experiment-card-wrapper");

        cards.forEach((card, index) => {
            card.classList.remove('active', 'above', 'below', 'hidden');

            if (index === this.currentExperimentGroupCard) {
                card.classList.add('active');
            } else if (index === (this.currentExperimentGroupCard - 1 + cards.length) % cards.length) {
                card.classList.add('above');
            } else if (index === (this.currentExperimentGroupCard + 1) % cards.length) {
                card.classList.add('below');
            } else {
                card.classList.add('hidden');
            }
        });
        this.updateCarouselHeight();
    }

    /**
     * Initialize the carousel with the first experiment group card.
     */
    initializeCarousel() {
        // ExperimentGroup Card Carousel Control
        const cards = document.querySelectorAll(".experiment-card-wrapper");
        if (cards.length === 0) return;

        this.currentExperimentGroupCard = 0;
        this.updateDatasetSelectionForCurrentCard();
        this.updateCarousel();

        setTimeout(() => this.updateHomeTables(), 100);
    }

    /**
     * Navigate the carousel to the next or previous card.
     * @param {number} direction - The direction to navigate. 1 for next, -1 for previous.
     */
    navigateCards(direction) {
        const cards = document.querySelectorAll(".experiment-card-wrapper");
        if (cards.length === 0) return;
        
        this.currentExperimentGroupCard = (this.currentExperimentGroupCard + direction + cards.length) % cards.length;
        this.updateDatasetSelectionForCurrentCard();
        this.updateCarousel();
        this.updateHomeTables();
    }

    /**
     * Update the dataset selection for the current card.
     */
    updateDatasetSelectionForCurrentCard() {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup) return;
        const datasetID = experimentGroup.name + "_" + experimentGroup.datasets[0];
        this.selectedDataset = this.reportData.datasets[datasetID];
        this.selectedSplit = 0;
    }

    /**
     * Update the height of the carousel based on the active card.
     */
    updateCarouselHeight() {
        const activeCard = document.querySelector('.experiment-card-wrapper.active');
        const viewport = document.querySelector('.cards-viewport');
        const track = document.querySelector('.cards-track');
        
        if (activeCard && viewport && track) {
            setTimeout(() => {
                const cardContent = activeCard.querySelector('.experiment-group-card');
                if (cardContent) {
                    const cardHeight = cardContent.scrollHeight;
                    const totalHeight = Math.max(cardHeight + 25, 400);                    
                    viewport.style.height = totalHeight + 'px';
                    track.style.height = totalHeight + 'px';
                }
            }, 50);
        }
    }

    /**
     * Initialize the experiment page.
     * @param {string} experimentId - The ID of the experiment.
     */
    initializeExperimentPage(experimentId) {
        const experiment = this.reportData.experiments[experimentId];
        if (!experiment) return;

        this.currentExperimentId = experimentId;
        this.selectedAlgorithmIndex = 0;
        this.selectedSplitIndex = 0;
        this.selectedTableIndex = 0;
        this.selectedPlotIndex = 0;

        const dataset = this.reportData.datasets[experiment.dataset];
        this.currentDatasetSplits = dataset ? dataset.splits : ['split_0'];

        setTimeout(() => {
            this.setupExperimentInteractivity();
            this.updateExperimentSummary();
            this.updateExperimentTables();
            this.updateExperimentPlots();
        }, 50);
    }

    /**
     * Setup the interactivity for the experiment page.
     */
    setupExperimentInteractivity() {
        this.setupAlgorithmSelector();
        this.setupSplitSelector();
        this.setupTableSelector();
        this.setupPlotSelector();
        this.setupExpandCollapseButtons();        
        this.updateAlgorithmSelection();
        this.updateSplitSelection();
        this.updateTableSelection();
        this.updatePlotSelection();
    }

    /**
     * Setup the expand/collapse buttons for the experiment page.
     */
    setupExpandCollapseButtons() {
        const tablesExpandButton = document.querySelector('.experiment-tables .expand-container');
        const plotsExpandButton = document.querySelector('.experiment-plots .expand-container');
        
        if (tablesExpandButton) {
            tablesExpandButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTableExpansion();
            });
        }
        
        if (plotsExpandButton) {
            plotsExpandButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.togglePlotExpansion();
            });
        }
    }

    /**
     * Toggle the table card expansion.
     */
    toggleTableExpansion() {
        const experimentRight = document.querySelector('.experiment-right');
        const tablesCheckbox = document.querySelector('.experiment-tables .expand-checkbox');
        const plotsCheckbox = document.querySelector('.experiment-plots .expand-checkbox');
        
        if (!experimentRight || !tablesCheckbox) return;

        const isTablesExpanded = experimentRight.classList.contains('tables-expanded');
        
        experimentRight.classList.remove('tables-expanded', 'plots-expanded');
        
        if (isTablesExpanded) {
            tablesCheckbox.checked = false;
            if (plotsCheckbox) plotsCheckbox.checked = false;
        } else {
            experimentRight.classList.add('tables-expanded');
            tablesCheckbox.checked = true;
            if (plotsCheckbox) plotsCheckbox.checked = false;
        }
    }

    /**
     * Toggle the plot card expansion.
     */
    togglePlotExpansion() {
        const experimentRight = document.querySelector('.experiment-right');
        const tablesCheckbox = document.querySelector('.experiment-tables .expand-checkbox');
        const plotsCheckbox = document.querySelector('.experiment-plots .expand-checkbox');
        
        if (!experimentRight || !plotsCheckbox) return;

        const isPlotsExpanded = experimentRight.classList.contains('plots-expanded');
        
        experimentRight.classList.remove('tables-expanded', 'plots-expanded');
        
        if (isPlotsExpanded) {
            plotsCheckbox.checked = false;
            if (tablesCheckbox) tablesCheckbox.checked = false;
        } else {
            experimentRight.classList.add('plots-expanded');
            plotsCheckbox.checked = true;
            if (tablesCheckbox) tablesCheckbox.checked = false;
        }
    }

    /**
     * Setup the algorithm selector for the experiment page if multiple algorithms are present.
     */
    setupAlgorithmSelector() {
        const algorithmNav = document.querySelector('.algorithm-nav');
        if (!algorithmNav) return;

        const algorithmItems = algorithmNav.querySelectorAll('.algorithm-text');
        algorithmItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectAlgorithm(index);
            });
        });
    }

    /**
     * Setup the split selector for the experiment page.
     */
    setupSplitSelector() {
        const splitsNav = document.querySelector('.splits-nav');
        if (!splitsNav) return;

        const splitItems = splitsNav.querySelectorAll('.split-text');
        splitItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectExperimentSplit(index);
            });
        });
    }

    /**
     * Setup the table selector for the experiment page.
     */
    setupTableSelector() {
        const tablesList = document.querySelector('.tables-list');
        if (!tablesList) return;

        const tableItems = tablesList.querySelectorAll('.table-name');
        tableItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectTable(index);
            });
        });
    }

    /**
     * Setup the plot selector for the experiment page.
     */
    setupPlotSelector() {
        const plotsList = document.querySelector('.plots-list');
        if (!plotsList) return;

        const plotItems = plotsList.querySelectorAll('.plot-name');
        plotItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectPlot(index);
            });
        });
    }

    /**
     * Select an algorithm on the experiment page.
     * @param {number} algorithmIndex - The index of the algorithm.
     */
    selectAlgorithm(algorithmIndex) {
        this.selectedAlgorithmIndex = algorithmIndex;
        this.updateAlgorithmSelection();        
        this.updateExperimentSummary();
        this.updateExperimentTables();
        this.updateExperimentPlots();
    }

    /**
     * Select a split on the experiment page.
     * @param {number} splitIndex - The index of the split.
     */
    selectExperimentSplit(splitIndex) {
        this.selectedSplitIndex = splitIndex;        
        this.updateSplitSelection();        
        this.updateExperimentSummary();
        this.updateExperimentTables();
        this.updateExperimentPlots();

        const breadcrumb = document.getElementById('experiment-breadcrumb');
        if (breadcrumb) {
            const currentText = breadcrumb.textContent;
            const updatedText = currentText.replace(/split_\d+/, `split_${splitIndex}`);
            breadcrumb.textContent = updatedText;
        }
    }

    /**
     * Select a table on the experiment page.
     * @param {number} tableIndex - The index of the table.
     */
    selectTable(tableIndex) {
        this.selectedTableIndex = tableIndex;
        this.updateTableSelection();        
        this.updateSelectedTable();
    }

    /**
     * Select a plot on the experiment page.
     * @param {number} plotIndex - The index of the plot.
     */
    selectPlot(plotIndex) {
        this.selectedPlotIndex = plotIndex;
        this.updatePlotSelection();        
        this.updateSelectedPlot();
    }

    /**
     * Update the algorithm selection on the experiment page.
     */
    updateAlgorithmSelection() {
        const algorithmNav = document.querySelector('.algorithm-nav');
        if (!algorithmNav) return;

        const algorithmItems = algorithmNav.querySelectorAll('.algorithm-text');
        algorithmItems.forEach((item, index) => {
            if (index === this.selectedAlgorithmIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the split selection on the experiment page.
     */
    updateSplitSelection() {
        const splitsNav = document.querySelector('.splits-nav');
        if (!splitsNav) return;

        const splitItems = splitsNav.querySelectorAll('.split-text');
        splitItems.forEach((item, index) => {
            if (index === this.selectedSplitIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the table selection on the experiment page.
     */
    updateTableSelection() {
        const tablesList = document.querySelector('.tables-list');
        if (!tablesList) return;

        const tableItems = tablesList.querySelectorAll('.table-name');
        tableItems.forEach((item, index) => {
            if (index === this.selectedTableIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the plot selection on the experiment page.
     */
    updatePlotSelection() {
        const plotsList = document.querySelector('.plots-list');
        if (!plotsList) return;

        const plotItems = plotsList.querySelectorAll('.plot-name');
        plotItems.forEach((item, index) => {
            if (index === this.selectedPlotIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the experiment summary on the experiment page.
     */
    updateExperimentSummary() {
        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment) return;

        this.updateExperimentTitle(experiment);
        this.updateTunedHyperparameters(experiment);
        this.updateHyperparameterGrid(experiment);
    }

    /**
     * Update the experiment title on the experiment page.
     * @param {Object} experiment - The experiment object.
     */
    updateExperimentTitle(experiment) {
        const titleElement = document.querySelector('.experiment-title');
        if (!titleElement) return;

        const selectedAlgorithm = experiment.algorithm[this.selectedAlgorithmIndex];
        
        if (experiment.algorithm.length > 1) {
            titleElement.textContent = `Experiment: ${selectedAlgorithm}`;
        } else {
            titleElement.textContent = selectedAlgorithm || experiment.ID;
        }
    }

    /**
     * Update the tuned hyperparameters on the experiment page.
     * @param {Object} experiment - The experiment object.
     */
    updateTunedHyperparameters(experiment) {
        const tbody = document.querySelector('.tuned-hyperparams-body');
        const message = document.querySelector('.tuned-hyperparams-message');
        const table = document.querySelector('.tuned-hyperparams-table');
        
        if (!tbody || !message || !table) return;

        tbody.innerHTML = '';
        
        if (experiment.tuned_params && Object.keys(experiment.tuned_params).length > 0) {
            message.style.display = 'none';
            table.style.display = 'table';
            
            Object.entries(experiment.tuned_params).forEach(([param, value]) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${param}</td>
                    <td>${value}</td>
                `;
                tbody.appendChild(row);
            });
        } else {
            message.style.display = 'block';
            table.style.display = 'none';
        }
    }

    /**
     * Update the hyperparameter grid on the experiment page.
     * @param {Object} experiment - The experiment object.
     */
    updateHyperparameterGrid(experiment) {
        const tbody = document.querySelector('.hyperparam-grid-body');
        const message = document.querySelector('.hyperparam-grid-message');
        const table = document.querySelector('.hyperparam-grid-table');
        
        if (!tbody || !message || !table) return;

        tbody.innerHTML = '';
        
        if (experiment.hyperparam_grid && Object.keys(experiment.hyperparam_grid).length > 0) {
            message.style.display = 'none';
            table.style.display = 'table';
            
            Object.entries(experiment.hyperparam_grid).forEach(([param, values]) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${param}</td>
                    <td>${values}</td>
                `;
                tbody.appendChild(row);
            });
        } else {
            message.style.display = 'block';
            table.style.display = 'none';
        }
    }

    /**
     * Update the tables on the experiment page.
     */
    updateExperimentTables() {
        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment) return;

        this.renderFilteredTables(experiment);
        
        const maxTableIndex = (experiment.tables || []).length - 1;
        if (this.selectedTableIndex > maxTableIndex) {
            this.selectedTableIndex = Math.max(0, maxTableIndex);
        }
        
        this.updateTableSelection();        
        this.updateSelectedTable();
    }

    /**
     * Update the plots on the experiment page.
     */
    updateExperimentPlots() {
        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment) return;

        this.renderFilteredPlots(experiment);
        
        const maxPlotIndex = (experiment.plots || []).length - 1;
        if (this.selectedPlotIndex > maxPlotIndex) {
            this.selectedPlotIndex = Math.max(0, maxPlotIndex);
        }
        
        this.updatePlotSelection();        
        this.updateSelectedPlot();
    }

    /**
     * Render the filtered tables on the experiment page.
     * @param {Object} experiment - The experiment object.
     */
    renderFilteredTables(experiment) {
        const tablesList = document.querySelector('.tables-list');
        if (!tablesList) return;

        tablesList.innerHTML = '';

        const tables = experiment.tables || [];

        tables.forEach((tableData, index) => {
            const tableName = document.createElement('div');
            tableName.textContent = tableData.name || `Table ${index + 1}`;
            tableName.className = `table-name ${index === this.selectedTableIndex ? 'selected' : ''}`;
            tableName.dataset.tableIndex = index;
            
            tableName.addEventListener('click', () => {
                this.selectTable(index);
            });
            
            tablesList.appendChild(tableName);
        });
    }

    /**
     * Render the filtered plots on the experiment page.
     * @param {Object} experiment - The experiment object.
     */
    renderFilteredPlots(experiment) {
        const plotsList = document.querySelector('.plots-list');
        if (!plotsList) return;

        plotsList.innerHTML = '';

        const plots = experiment.plots || [];

        plots.forEach((plotData, index) => {
            const plotName = document.createElement('div');
            plotName.textContent = plotData.name || `Plot ${index + 1}`;
            plotName.className = `plot-name ${index === this.selectedPlotIndex ? 'selected' : ''}`;
            plotName.dataset.plotIndex = index;
            
            plotName.addEventListener('click', () => {
                this.selectPlot(index);
            });
            
            plotsList.appendChild(plotName);
        });
    }

    /**
     * Filter the table data for the current split and algorithm selection.
     * @param {Object} originalTableData - The original table data.
     * @returns {Object} - The filtered table data.
     */
    filterTableDataForCurrentSelection(originalTableData) {
        if (!originalTableData) return originalTableData;
        
        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment) return originalTableData;

        const splitDisplayName = `Split ${this.selectedSplitIndex}`;
        const currentAlgorithm = experiment.algorithm[this.selectedAlgorithmIndex];
        const splitColumnIndex = originalTableData.columns.findIndex(col => 
            col.toLowerCase().includes('split')
        );

        let filteredRows = originalTableData.rows;
        if (splitColumnIndex !== -1) {
            filteredRows = filteredRows.filter(row => 
                row[splitColumnIndex] === splitDisplayName
            );
        }

        if (experiment.algorithm.length > 1) {
            const algorithmColumnIndex = originalTableData.columns.findIndex(col => 
                col.toLowerCase().includes('algorithm')
            );
            
            if (algorithmColumnIndex !== -1) {
                filteredRows = filteredRows.filter(row => 
                    row[algorithmColumnIndex] === currentAlgorithm
                );
            }
        }

        return {
            ...originalTableData,
            rows: filteredRows,
            description: `${originalTableData.description} (Split ${this.selectedSplitIndex}${experiment.algorithm.length > 1 ? `, ${currentAlgorithm}` : ''})`
        };
    }

    /**
     * Update the selected table on the experiment page.
     */
    updateSelectedTable() {
        const tablesContent = document.querySelector('.tables-content');
        if (!tablesContent) return;

        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment || !experiment.tables) return;

        tablesContent.innerHTML = '';

        const selectedTable = experiment.tables[this.selectedTableIndex];
        if (selectedTable) {
            try {
                const filteredTableData = this.filterTableDataForCurrentSelection(selectedTable);
                const tableRenderer = new TableRenderer(filteredTableData);
                const tableElement = tableRenderer.render();
                tablesContent.appendChild(tableElement);
            } catch (error) {
                console.error('Error rendering table:', error);
                tablesContent.innerHTML = '<div class="error-message">Error loading table</div>';
            }
        } else {
            tablesContent.innerHTML = '<div class="no-data-message">No table selected</div>';
        }
    }

    /**
     * Filter the plot data for the current selection.
     * @param {Object} originalPlotData - The original plot data.
     * @returns {Object} - The filtered plot data.
     */
    filterPlotDataForCurrentSelection(originalPlotData) {
        if (!originalPlotData) return originalPlotData;
        
        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment) return originalPlotData;

        const currentAlgorithm = experiment.algorithm[this.selectedAlgorithmIndex];
        let updatedDescription = originalPlotData.description;
        if (experiment.algorithm.length > 1) {
            updatedDescription = `${originalPlotData.description} (${currentAlgorithm}, Split ${this.selectedSplitIndex})`;
        } else {
            updatedDescription = `${originalPlotData.description} (Split ${this.selectedSplitIndex})`;
        }

        return {
            ...originalPlotData,
            description: updatedDescription
        };
    }

    /**
     * Update the selected plot on the experiment page.
     */
    updateSelectedPlot() {
        const plotsContent = document.querySelector('.plots-content');
        if (!plotsContent) return;

        const experiment = this.reportData.experiments[this.currentExperimentId];
        if (!experiment || !experiment.plots) return;

        plotsContent.innerHTML = '';

        const selectedPlot = experiment.plots[this.selectedPlotIndex];
        if (selectedPlot) {
            try {
                const filteredPlotData = this.filterPlotDataForCurrentSelection(selectedPlot);
                const plotRenderer = new PlotRenderer(filteredPlotData);
                const plotElement = plotRenderer.render();
                plotsContent.appendChild(plotElement);
            } catch (error) {
                console.error('Error rendering plot:', error);
                plotsContent.innerHTML = '<div class="error-message">Error loading plot</div>';
            }
        } else {
            plotsContent.innerHTML = '<div class="no-data-message">No plot selected</div>';
        }
    }

    /**
     * Show the experiment navigation when on the experiment page.
     * @param {string} experimentId - The ID of the experiment.
     */
    showExperimentNavigation(experimentId) {
        const navContainer = document.getElementById('experiment-navigation');
        if (!navContainer) return;

        const experiment = this.reportData.experiments[experimentId];
        if (!experiment) return;

        const experimentGroup = this.findExperimentGroup(experimentId);
        if (!experimentGroup) return;

        const datasetExperiments = this.getDatasetExperiments(experimentGroup, experiment.dataset);
        const currentIndex = datasetExperiments.indexOf(experimentId);

        this.populateExperimentNavigation(experiment, experimentGroup);
        this.setupExperimentNavigationButtons(datasetExperiments, currentIndex);
        navContainer.style.display = 'block';
    }

    /**
     * Hide the experiment navigation when not on the experiment page.
     */
    hideExperimentNavigation() {
        const navContainer = document.getElementById('experiment-navigation');
        if (navContainer) {
            navContainer.style.display = 'none';
        }
    }

    /**
     * Find the experiment group for the given experiment ID.
     * @param {string} experimentId - The ID of the experiment.
     * @returns {Object} - The experiment group.
     */
    findExperimentGroup(experimentId) {
        return this.reportData.experiment_groups.find(group => 
            group.experiments.includes(experimentId)
        );
    }

    /**
     * Get the experiments for the given dataset ID.
     * @param {Object} experimentGroup - The experiment group.
     * @param {string} datasetId - The ID of the dataset.
     * @returns {Array} - The experiments.
     */
    getDatasetExperiments(experimentGroup, datasetId) {
        return experimentGroup.experiments
            .filter(expId => {
                const exp = this.reportData.experiments[expId];
                return exp && exp.dataset === datasetId;
            })
            .sort((a, b) => a.localeCompare(b));
    }

    /**
     * Populate the experiment navigation.
     * @param {Object} experiment - The experiment object.
     * @param {Object} experimentGroup - The experiment group.
     */
    populateExperimentNavigation(experiment, experimentGroup) {
        document.getElementById('current-experiment-group').textContent = experimentGroup.name;        
        const dataset = this.reportData.datasets[experiment.dataset];
        
        let datasetDisplayName;
        if (dataset) {
            const groupPrefix = experimentGroup.name + '_';
            if (experiment.dataset.startsWith(groupPrefix)) {
                datasetDisplayName = experiment.dataset.substring(groupPrefix.length);
            } else {
                datasetDisplayName = experiment.dataset;
            }
        } else {
            datasetDisplayName = 'Unknown Dataset';
        }
        
        document.getElementById('current-dataset-name').textContent = datasetDisplayName;

        const breadcrumb = document.getElementById('experiment-breadcrumb');
        const groupSlug = experimentGroup.name.toLowerCase().replace(/\s+/g, '-');
        const datasetSlug = experiment.dataset.toLowerCase().replace(/\s+/g, '-');
        
        const experimentDisplayName = experiment.algorithm && experiment.algorithm.length > 0 
            ? experiment.algorithm.join('-').toLowerCase().replace(/\s+/g, '-')
            : experiment.ID.toLowerCase().replace(/\s+/g, '-');
        
        breadcrumb.textContent = `${groupSlug}/${datasetSlug}/split_${this.selectedSplitIndex}/${experimentDisplayName}`;
    }

    /**
     * Setup the experiment navigation buttons.
     * @param {Array} datasetExperiments - The experiments for the given dataset.
     * @param {number} currentIndex - The current index of the experiment.
     */
    setupExperimentNavigationButtons(datasetExperiments, currentIndex) {
        const prevBtn = document.getElementById('prev-experiment-btn');
        prevBtn.replaceWith(prevBtn.cloneNode(true));

        const nextBtn = document.getElementById('next-experiment-btn');
        nextBtn.replaceWith(nextBtn.cloneNode(true));
        
        const newPrevBtn = document.getElementById('prev-experiment-btn');
        const newNextBtn = document.getElementById('next-experiment-btn');

        if (currentIndex > 0) {
            newPrevBtn.disabled = false;
            newPrevBtn.addEventListener('click', () => {
                const prevExperimentId = datasetExperiments[currentIndex - 1];
                this.showPage('experiment', prevExperimentId);
            });
        } else {
            newPrevBtn.disabled = true;
        }

        if (currentIndex < datasetExperiments.length - 1) {
            newNextBtn.disabled = false;
            newNextBtn.addEventListener('click', () => {
                const nextExperimentId = datasetExperiments[currentIndex + 1];
                this.showPage('experiment', nextExperimentId);
            });
        } else {
            newNextBtn.disabled = true;
        }
    }

    /**
     * Initialize the dataset page.
     */
    initializeDatasetPage() {
        if (!this.selectedDataset) return;

        this.currentDatasetId = this.selectedDataset.ID;
        this.selectedDatasetSplitIndex = 0;
        this.selectedFeatureIndex = 0;

        setTimeout(() => {
            this.setupDatasetInteractivity();
            this.updateDatasetSummary();
            this.updateDatasetDistribution();
        }, 50);
    }

    /**
     * Setup the interactivity for the dataset page.
     */
    setupDatasetInteractivity() {
        this.setupDatasetSplitSelector();
        this.setupDatasetFeatureSelector();
        this.setupDatasetCollapsibleSections();
        this.setupCorrelationModal();
        this.updateDatasetSplitSelection();
        this.updateDatasetFeatureSelection();
    }

    /**
     * Setup the collapsible sections for the dataset page.
     */
    setupDatasetCollapsibleSections() {
        const dmHeader = document.querySelector('.collapsible-header');
        const dmContent = document.querySelector('.collapsible-content');
        const collapseIcon = document.querySelector('.collapse-icon');
        
        if (dmHeader && dmContent && collapseIcon) {
            dmHeader.addEventListener('click', () => {
                const isCollapsed = dmContent.style.display === 'none';
                dmContent.style.display = isCollapsed ? 'block' : 'none';
                collapseIcon.textContent = isCollapsed ? '▼' : '▲';
            });
        }
    }

    /**
     * Setup the data split selector for the dataset page.
     */
    setupDatasetSplitSelector() {
        const splitsNav = document.querySelector('.dataset-splits .splits-nav');
        if (!splitsNav) return;

        const splitItems = splitsNav.querySelectorAll('.split-text');
        splitItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectDatasetSplit(index);
            });
        });
    }

    /**
     * Setup the feature selector for the dataset page.
     */
    setupDatasetFeatureSelector() {
        const featuresNav = document.querySelector('.dataset-features .features-nav');
        if (!featuresNav) return;

        const featureItems = featuresNav.querySelectorAll('.feature-text');
        featureItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.selectDatasetFeature(index);
            });
        });
    }

    /**
     * Setup the correlation modal for the dataset page.
     */
    setupCorrelationModal() {
        const correlationContent = document.querySelector('.correlation-content');
        const modal = document.getElementById('correlationModal');
        const modalClose = document.getElementById('correlationModalClose');
        const modalPlot = document.getElementById('correlationModalPlot');
        
        if (!correlationContent || !modal || !modalClose || !modalPlot) return;
        
        // Open modal on correlation matrix click
        correlationContent.addEventListener('click', () => {
            this.openCorrelationModal();
        });
        
        // Close modal on close button click
        modalClose.addEventListener('click', () => {
            this.closeCorrelationModal();
        });
        
        // Close modal on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeCorrelationModal();
            }
        });
        
        // Close modal on ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('show')) {
                this.closeCorrelationModal();
            }
        });
    }

    /**
     * Open the correlation modal.
     */
    openCorrelationModal() {
        const modal = document.getElementById('correlationModal');
        const modalPlot = document.getElementById('correlationModalPlot');
        const modalTitle = document.querySelector('.correlation-modal-title');
        const correlationPlot = document.querySelector('.correlation-plot');
        
        if (!modal || !modalPlot || !correlationPlot) return;
        
        // Update modal title with current dataset and split info
        if (modalTitle && this.selectedDataset) {
            const splitText = `Split ${this.selectedDatasetSplitIndex}`;
            modalTitle.textContent = `Correlation Matrix - ${this.selectedDataset.ID} (${splitText})`;
        }
        
        // Copy the correlation matrix content to modal
        modalPlot.innerHTML = correlationPlot.innerHTML;
        
        // Add blur to background
        document.body.classList.add('modal-open');
        
        // Show modal with animation
        modal.classList.add('show');
        
        // Prevent body scroll
        document.body.style.overflow = 'hidden';
    }

    /**
     * Close the correlation modal.
     */
    closeCorrelationModal() {
        const modal = document.getElementById('correlationModal');
        
        if (!modal) return;
        
        // Remove blur from background
        document.body.classList.remove('modal-open');
        
        // Hide modal with animation
        modal.classList.remove('show');
        
        // Restore body scroll
        document.body.style.overflow = '';
        
        // Clear modal content after animation
        setTimeout(() => {
            const modalPlot = document.getElementById('correlationModalPlot');
            if (modalPlot) {
                modalPlot.innerHTML = '';
            }
        }, 300);
    }

    /**
     * Select a data split on the dataset page.
     * @param {number} splitIndex - The index of the split.
     */
    selectDatasetSplit(splitIndex) {
        this.selectedDatasetSplitIndex = splitIndex;
        this.updateDatasetSplitSelection();        
        this.updateDatasetSummary();
        this.updateDatasetDistribution();
    }

    /**
     * Select a feature on the dataset page.
     * @param {number} featureIndex - The index of the feature.
     */
    selectDatasetFeature(featureIndex) {
        this.selectedFeatureIndex = featureIndex;        
        this.updateDatasetFeatureSelection();        
        this.updateDatasetDistribution();
    }

    /**
     * Update the split selection on the dataset page.
     */
    updateDatasetSplitSelection() {
        const splitsNav = document.querySelector('.dataset-splits .splits-nav');
        if (!splitsNav) return;

        const splitItems = splitsNav.querySelectorAll('.split-text');
        splitItems.forEach((item, index) => {
            if (index === this.selectedDatasetSplitIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the selected feature on the dataset page.
     */
    updateDatasetFeatureSelection() {
        const featuresNav = document.querySelector('.dataset-features .features-nav');
        if (!featuresNav) return;

        const featureItems = featuresNav.querySelectorAll('.feature-text');
        featureItems.forEach((item, index) => {
            if (index === this.selectedFeatureIndex) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
    }

    /**
     * Update the dataset summary on the dataset page.
     */
    updateDatasetSummary() {
        if (!this.selectedDataset) return;
        this.updateDatasetInfoTables();
        this.updateDatasetCorrelationMatrix();
    }

    /**
     * Update the dataset info tables on the dataset page.
     */
    updateDatasetInfoTables() {
        const datasetInfoContent = document.querySelector('.dataset-info-content');
        datasetInfoContent.innerHTML = '';
       
        const targetInfoContent = document.querySelector('.target-info-content');
        targetInfoContent.innerHTML = '';
        
        if (!datasetInfoContent || !targetInfoContent || !this.selectedDataset) return;

        const currentSplitId = this.selectedDataset.splits[this.selectedDatasetSplitIndex];
        const currentSplitSizes = this.selectedDataset.split_sizes[currentSplitId] || {};
        const currentTargetStats = this.selectedDataset.split_target_stats[currentSplitId] || {};
        
        
        if (currentSplitSizes.features && currentSplitSizes.total_obs) {
            datasetInfoContent.innerHTML = `
                <div class="info-item-compact">
                    <span class="info-label-compact">Features:</span>
                    <span class="info-value-compact">${currentSplitSizes.features}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Total Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.total_obs}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Train Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.train_obs || 'N/A'}</span>
                </div>
                <div class="info-item-compact">
                    <span class="info-label-compact">Test Observations:</span>
                    <span class="info-value-compact">${currentSplitSizes.test_obs || 'N/A'}</span>
                </div>
            `;
        } else {
            datasetInfoContent.innerHTML = '<div class="info-item-compact">No dataset information available</div>';
        }
        
        // Populate target feature stats
        if (Object.keys(currentTargetStats).length > 0) {
            Object.entries(currentTargetStats).forEach(([stat, value]) => {
                const statItem = document.createElement('div');
                statItem.className = 'info-item-compact';
                statItem.innerHTML = `
                    <span class="info-label-compact">${stat.charAt(0).toUpperCase() + stat.slice(1)}:</span>
                    <span class="info-value-compact">${value}</span>
                `;
                targetInfoContent.appendChild(statItem);
            });
        } else {
            targetInfoContent.innerHTML = '<div class="info-item-compact">No target stats available</div>';
        }
    }

    /**
     * Update the correlation matrix on the dataset page.
     */
    updateDatasetCorrelationMatrix() {
        const corrPlot = document.querySelector('.correlation-plot');
        if (!corrPlot || !this.selectedDataset) return;
        
        corrPlot.innerHTML = '';
        
        const currentSplitId = this.selectedDataset.splits[this.selectedDatasetSplitIndex];
        const currentCorrMatrix = this.selectedDataset.split_corr_matrices[currentSplitId];
        
        if (currentCorrMatrix && currentCorrMatrix.image) {
            corrPlot.innerHTML = currentCorrMatrix.image;
        } else {
            corrPlot.innerHTML = '<div class="correlation-placeholder">Correlation Matrix</div>';
        }
        
        const modal = document.getElementById('correlationModal');
        const modalPlot = document.getElementById('correlationModalPlot');
        if (modal && modal.classList.contains('show') && modalPlot) {
            modalPlot.innerHTML = corrPlot.innerHTML;
            
            const modalTitle = document.querySelector('.correlation-modal-title');
            if (modalTitle) {
                const splitText = `Split ${this.selectedDatasetSplitIndex}`;
                modalTitle.textContent = `Correlation Matrix - ${this.selectedDataset.ID} (${splitText})`;
            }
        }
    }

    /**
     * Update the dataset distribution on the dataset page.
     */
    updateDatasetDistribution() {
        const tableContainer = document.querySelector('.distribution-table-content');
        tableContainer.innerHTML = '';
       
        const plotContainer = document.querySelector('.distribution-plot-content');
        plotContainer.innerHTML = '';

        const tableTitle = document.querySelector('.distribution-table-title');
        
        if (!tableContainer || !plotContainer || !tableTitle || !this.selectedDataset) return;
        
        
        const currentSplitId = this.selectedDataset.splits[this.selectedDatasetSplitIndex];
        const currentFeatureDistributions = this.selectedDataset.split_feature_distributions[currentSplitId];
        
        if (currentFeatureDistributions && currentFeatureDistributions.length > 0) {
            const selectedDistribution = currentFeatureDistributions[this.selectedFeatureIndex] || currentFeatureDistributions[0];
            
            tableTitle.textContent = selectedDistribution.tables[0].name;
            
            for (const table of selectedDistribution.tables) {
                const tableRenderer = new TableRenderer(table);
                const tableElement = tableRenderer.render();
                tableContainer.appendChild(tableElement);
            }
            
            const plotRenderer = new PlotRenderer(selectedDistribution.plot);
            const plotElement = plotRenderer.render();
            plotContainer.appendChild(plotElement);
        } else {
            tableTitle.textContent = 'Feature Distribution';
            
            const noDataMessage = document.createElement('p');
            noDataMessage.textContent = 'No feature distribution data available';
            noDataMessage.className = 'no-data-message';
            tableContainer.appendChild(noDataMessage);
            
            const plotPlaceholder = document.createElement('div');
            plotPlaceholder.className = 'plot-placeholder';
            plotPlaceholder.textContent = 'Distribution Plot';
            plotContainer.appendChild(plotPlaceholder);
        }
    }    
}
