class ExperimentGroupCardRenderer {
    constructor(cardData, cardIndex) {
        this.cardData = cardData;
        this.cardIndex = cardIndex;
    }

    render() {
        const wrapper = document.createElement('div');
        wrapper.className = 'experiment-card-wrapper';
        wrapper.setAttribute('group-card-index', this.cardIndex);

        const template = document.getElementById('experiment-group-card-template').content.cloneNode(true);
        this.renderTitle(template);
        this.renderDescription(template);
        this.renderDatasets(template);
        this.renderExperiments(template);
        this.renderDataSplits(template);

        wrapper.appendChild(template);
        return wrapper;
    }

    renderTitle(template) {
        const titleElement = template.querySelector('.card-title');
        titleElement.textContent = this.cardData.name;
    }

    renderDescription(template) {
        const descriptionElement = template.querySelector('.card-description');
        if (this.cardData.description) {
            descriptionElement.textContent = this.cardData.description;
        }
    }

    renderDatasets(template) {
        const datasetContainer = template.querySelector('.group-dataset-container');
        
        if (!this.cardData.datasets || this.cardData.datasets.length === 0) {
            return;
        }

        this.cardData.datasets.forEach((datasetID, index) => {
            const datasetName = this.formatDatasetName(datasetID);
            const datasetLink = document.createElement('a');
            datasetLink.href = '#';
            datasetLink.className = 'dataset-name';
            datasetLink.textContent = datasetName;

            if (index === 0) {
                datasetLink.classList.add('selected');
            }
            
            datasetLink.setAttribute('onclick', `window.app.selectDataset(this, '${datasetName}', ${this.cardIndex})`);
            datasetContainer.appendChild(datasetLink);
        });
        
        if (this.cardData.datasets.length > 0) {
            datasetContainer.setAttribute('data-selected-dataset', this.cardData.datasets[0]);
        }
    }

    renderExperiments(template) {
        const experimentList = template.querySelector('.experiment-list');
        
        if (!this.cardData.experiments || this.cardData.experiments.length === 0) {
            return;
        }

        this.cardData.experiments.forEach(experimentID => {
            const experimentData = window.app.reportData.experiments[experimentID]
            const listItem = document.createElement('li');

            const experimentAnchor = document.createElement('a');
            experimentAnchor.href = '#';
            experimentAnchor.setAttribute('page-type', 'experiment');
            experimentAnchor.setAttribute('page-data', JSON.stringify(experimentData));
            experimentAnchor.className = 'experiment-link';
            experimentAnchor.textContent = experimentID;
            
            listItem.appendChild(experimentAnchor);
            experimentList.appendChild(listItem);
        });
    }

    renderDataSplits(template) {
        const splitSelector = template.querySelector('.data-split-selector');
        
        if (!this.cardData.data_split_scores || Object.keys(this.cardData.data_split_scores).length === 0) {
            return;
        }

        const table = document.createElement('table');
        table.className = 'split-table';
        table.id = `split-table-${this.cardIndex}`;
        
        const firstDataset = this.cardData.datasets[0];
        const firstDatasetSplits = this.cardData.data_split_scores[firstDataset];
        
        // Get metric from first split (4th element in tuple)
        let metricName = "Score"; // Default fallback
        if (firstDatasetSplits && firstDatasetSplits.length > 0) {
            const firstSplit = firstDatasetSplits[0];
            if (firstSplit.length > 3) {
                metricName = firstSplit[3];
            }
        }
        
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Split</th>
                <th>Best Algorithm</th>
                <th>${metricName}</th>
            </tr>
        `;
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        table.appendChild(tbody);
        
        if (firstDatasetSplits) {
            firstDatasetSplits.forEach((split, index) => {
                const [splitName, algorithm, score] = split;
                
                const row = document.createElement('tr');
                row.className = 'split-row';
                row.setAttribute('data-split-index', index);
                if (index === 0) {
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
        
        splitSelector.appendChild(table);
    }

    formatDatasetName(datasetID) {
        const datasetName = datasetID.replace(`${this.cardData.name}_`, '');
        return datasetName;
    }
}
