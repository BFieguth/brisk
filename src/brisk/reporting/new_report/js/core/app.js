class App {
    constructor(reportData) {
        this.reportData = reportData;
        this.currentPage = 'home';
        this.pendingTableUpdate = null;
        this.isProcessingAnimation = false;

        //  Home page state
        this.currentExperimentGroupCard = 0;
        this.selectedDataset = null; // This must be as Dataset instance
        this.selectedSplit = 0;
        this.initalizeHomeSelections();
    }

    init() {
        this.initializeTheme();
        this.showPage('home');
        this.setupNavigation();
    }

    initalizeHomeSelections() {
        const experimentGroups = this.reportData.experiment_groups;
        if (experimentGroups.length > 0) {
            const firstGroup = experimentGroups[0];
            const datasetID = firstGroup.datasets[0];
            this.selectedDataset = this.reportData.datasets[datasetID];
        }
    }
    // Theme Management
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

    //  Client side page rendering
    showPage(pageType, pageData = null) {
        const mainContent = document.getElementById('main-content');
        this.currentPage = pageType;

        try {
            let content = this.renderPage(pageType, pageData);
            mainContent.innerHTML = content;

            if (pageType === 'home') {
                // Ensure DOM is ready first
                setTimeout(() => this.initializeCarousel(), 50);
            }
        } catch (e) {
            console.error('Error rendering page:', e);
            mainContent.innerHTML = '<div>Error rendering page: ' + e.message + '</div>';
        }
    }

    renderPage(pageType, pageData = null) {
        switch(pageType) {
            case 'home':
                return this.renderHomePage();
            case 'experiment':
                return this.renderExperimentPage(pageData);
            case 'dataset':
                return this.renderDatasetPage(pageData);
            default:
                return '<div>Renderer not found</div>';
        }
    }

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

    renderExperimentPage(experimentData) {
        const renderer = new ExperimentPageRenderer(experimentData);
        // const renderedElement = renderer.render();
        // const tempDiv = document.createElement('div');
        // tempDiv.appendChild(renderedElement);
        // return tempDiv.innerHTML;
        return renderer.render()
    }

    renderDatasetPage(datasetData) {
        const renderer = new DatasetPageRenderer(datasetData);
        // const renderedElement = renderer.render();
        // const tempDiv = document.createElement('div');
        // tempDiv.appendChild(renderedElement);
        // return tempDiv.innerHTML;
        return renderer.render()

    }

    //  Navigation
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

    //  Home Page Logic
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

        // Prevent default link behavior
        return false;
    }

    getCurrentExperimentGroup() {
        const experimentGroups = this.reportData.experiment_groups;
        if (experimentGroups.length > this.currentExperimentGroupCard) {
            return experimentGroups[this.currentExperimentGroupCard];
        }
        return null;
    }

    getCurrentSelectedTableData() {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup || !this.selectDataset) {
            return null;
        }
        const tableKey = `${this.selectedDataset.ID}_split_${this.selectedSplit}`;
        return experimentGroup.test_scores[tableKey] || null;
    }

    updateDataSplitsTable(cardIndex, datasetID) {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup) return;

        const splitTable = document.getElementById(`split-table-${cardIndex}`);

        console.log(datasetID)

        if (!splitTable || !experimentGroup.data_split_scores[datasetID]) return;

        const splitData = experimentGroup.data_split_scores[datasetID];

        console.log('selcted split', splitData)
        
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

    selectSplit(clickedRow, splitIndex) {
        const table = clickedRow.closest('table');
        const allRows = table.querySelectorAll('.split-row');
        allRows.forEach(row => row.classList.remove('selected'));

        clickedRow.classList.add('selected');
        this.selectedSplit = splitIndex;
        this.updateHomeTables();
    }

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

    initializeCarousel() {
        // ExperimentGroup Card Carousel Control
        const cards = document.querySelectorAll(".experiment-card-wrapper");
        if (cards.length === 0) return;

        this.currentExperimentGroupCard = 0;
        this.updateDatasetSelectionForCurrentCard();
        this.updateCarousel();

        setTimeout(() => this.updateHomeTables(), 100);
    }

    navigateCards(direction) {
        const cards = document.querySelectorAll(".experiment-card-wrapper");
        if (cards.length === 0) return;
        
        this.currentExperimentGroupCard = (this.currentExperimentGroupCard + direction + cards.length) % cards.length;
        this.updateDatasetSelectionForCurrentCard();
        this.updateCarousel();
        this.updateHomeTables();
    }

    updateDatasetSelectionForCurrentCard() {
        const experimentGroup = this.getCurrentExperimentGroup();
        if (!experimentGroup) return;
        const datasetID = experimentGroup.datasets[0];
        this.selectedDataset = this.reportData.datasets[datasetID];
        this.selectedSplit = 0;
    }

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
}
