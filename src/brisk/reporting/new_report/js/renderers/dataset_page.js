class DatasetPageRenderer {
    constructor(datasetData) {
        this.datasetData = datasetData;
    }

    render() {
        // Temporary content showing the clicked dataset name  
        return `
        <div class="page-content">
            <h1>Dataset Page</h1>
            <p>You clicked on dataset: <strong>${this.datasetData}</strong></p>
            <p>This is temporary content. The full dataset details will be implemented here.</p>
        </div>
    `;
    }
}