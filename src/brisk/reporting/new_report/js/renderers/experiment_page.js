class ExperimentPageRenderer {
    constructor(experimentData) {
        this.experimentData = experimentData;
    }

    render() {
        // Temporary content showing the clicked experiment name
        return `
        <div class="page-content">
            <h1>Experiment Page</h1>
            <p>You clicked on experiment: <strong>${this.experimentData}</strong></p>
            <p>This is temporary content. The full experiment details will be implemented here.</p>
        </div>
    `;
    }
}