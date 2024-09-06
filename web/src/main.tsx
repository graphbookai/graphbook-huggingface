import React from "react";
import { ExamplePanel } from "./ExamplePanel";
import { Pipeline } from "./Pipeline";

type GraphbookAPI = {
    useAPI: Function,
    useAPIMessage: Function,
    useAPINodeMessage: Function,
};

export function ExportSteps(graphbookAPI: GraphbookAPI) {
    return [{
        for: {
            name: "HuggingfacePipeline",
        },
        component: Pipeline,
    }];
}

export function ExportPanels(graphbookAPI: GraphbookAPI) {
    console.log(graphbookAPI);
    const { useAPIMessage } = graphbookAPI;
    return [{
        label: "Example Panel",
        children: <ExamplePanel useAPIMessage={useAPIMessage}/>,
    }];
}
