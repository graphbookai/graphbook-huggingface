import React, { useState, useCallback, useMemo } from 'react';
import { Flex, Input, Typography, Card, Select, Divider, Tag, Tooltip, theme } from 'antd';
import { DownloadOutlined, HeartOutlined, CopyOutlined } from '@ant-design/icons';

const bindDragData = (value: object, e: DragEvent) => {
    if (e) {
        e.dataTransfer?.setData('application/json', JSON.stringify(value));
    }
};
const { Text, Link } = Typography;
const { Search } = Input;

const SEARCH_TIMEOUT_DELAY = 1000;
const HF_HOST = "https://huggingface.co";
interface HFData {
    id: string;
    author: string;
    tags: string[];
    downloads: number;
    likes: number;
    idParts: string[];
}
interface HFModel extends HFData {
    pipelineTag: string;
};
interface HFDataset extends HFData { };

const tagStyle: React.CSSProperties = {
    maxWidth: '120px',
    padding: 4,
    margin: '0px 1px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    lineHeight: '1'
};

const selectStyle: React.CSSProperties = {
    padding: 2
};

const HFPipelineExtraParameters = {
    "batch_size": {
        "type": "number",
        "value": 8,
        "description": "The batch size for the pipeline",
    },
    "item_key": {
        "type": "string",
        "description": "The key in the input item to use as input. Should be reference a URL linking to an image, a base64 string, a local path, or a PIL image.",
        "value": ""
    },
    "device_id": {
        "type": "number",
        "value": 0,
        "description": "The GPU ID to use for the pipeline",
    },
    "fp16": {
        "type": "boolean",
        "value": false,
        "description": "Whether to use fp16",
    },
    "log_model_outputs": {
        "type": "boolean",
        "value": true,
        "description": "Whether to log the model outputs as JSON to the node UI",
    },
    "on_model_outputs": {
        "type": "resource",
        "description": "The function called when model outputs are received from the pipeline. By default, you may use AssignModelOutputToNotes.",
        "required": false,
        "value": null,
    },
    "kwargs": {
        "type": "dict",
        "description": "Additional keyword arguments to pass to the model pipeline",
        "required": false,
    }
};

const getHFPipelineNode = (modelId: string) => {
    return {
        type: 'step',
        data: {
            name: 'HuggingfacePipeline',
            label: 'HuggingfacePipeline',
            inputs: ["in"],
            outputs: ["out"],
            parameters: {
                model_id: {
                    type: "string",
                    value: modelId,
                    description: "The model ID to use for the pipeline",
                },
                ...HFPipelineExtraParameters
            }
        }
    };
};

const HFDatasetExtraParameters = {
    "split": {
        "type": "string",
        "value": "train",
        "description": "The split of the dataset to use",
    },
    "log_data": {
        "type": "boolean",
        "value": true,
        "description": "Whether to log the outputs as JSON to the node UI",
    },
    "image_columns": {
        "type": "list[string]",
        "description": "The columns in the dataset that contain images. This is to let Graphbook know how to display the images in the UI.",
        "required": false,
        "value": []
    },
    "kwargs": {
        "type": "dict",
        "description": "Additional keyword arguments to pass to the dataset",
        "required": false,
    }
};

const getHFDatasetNode = (datasetId: string) => {
    return {
        type: 'step',
        data: {
            name: 'HuggingfaceDataset',
            label: 'HuggingfaceDataset',
            inputs: [],
            outputs: ["out"],
            parameters: {
                dataset_id: {
                    type: "string",
                    value: datasetId,
                    description: "The dataset ID to use",
                },
                ...HFDatasetExtraParameters
            }
        }
    };
};

export function HF() {
    const [results, setResults] = useState<HFData[]>([]);
    const [searchTimeout, setSearchTimeout] = useState<NodeJS.Timeout>(null);
    const [searchType, setSearchType] = useState<string>('models');
    const [search, setSearch] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const loadModelList = useCallback(async (search, searchType) => {
        if (search.length === 0) {
            setResults([]);
            setIsLoading(false);
            return;
        }
        const path = `${HF_HOST}/api/${searchType}?search=${search}&limit=10&full=true`;
        try {
            const response = await fetch(path, {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
            });
            const data = await response.json();
            if (data) {
                let results = [];
                if (searchType === 'models') {
                    results = data.map((data: any) => {
                        return {
                            id: data.id,
                            author: data.author,
                            tags: data.tags,
                            downloads: data.downloads,
                            likes: data.likes,
                            idParts: data.id.split('/'),
                            pipelineTag: data.pipeline_tag,
                        };
                    });
                } else {
                    results = data.map((data: any) => {
                        return {
                            id: data.id,
                            author: data.author,
                            tags: data.tags,
                            downloads: data.downloads,
                            likes: data.likes,
                            idParts: data.id.split('/')
                        };
                    });
                }
                setResults(results);
                setIsLoading(false);
            }
        } catch (e) {
            console.error(e);
            setResults([]);
            setIsLoading(false);
        }
    }, []);

    const onSearchChange = useCallback((e: any) => {
        setSearch(e.target.value);
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }
        setIsLoading(true);

        const timeout = setTimeout(() => loadModelList(e.target.value, searchType), SEARCH_TIMEOUT_DELAY);
        setSearchTimeout(timeout);
    }, [searchTimeout, searchType]);

    const onSearchTypeChange = useCallback((value: string) => {
        setSearchType(value);
        if (searchTimeout) {
            clearTimeout(searchTimeout);
        }
        setIsLoading(true);

        setResults([]);
        loadModelList(search, value);
    }, [search]);

    return (
        <Flex vertical style={{ height: '100%', width: '100%' }}>
            <Search addonBefore={
                <Select dropdownStyle={selectStyle} defaultValue="models" onChange={onSearchTypeChange}>
                    <Select.Option value="models">Models</Select.Option>
                    <Select.Option value="datasets">Datasets</Select.Option>
                </Select>
            }
                style={{ marginBottom: 5 }} placeholder="Search" onChange={onSearchChange} loading={isLoading} />
            <Flex vertical style={{ overflowY: 'auto', paddingRight: 5 }}>
                {
                    results.map((result: HFData, i: number) => <ResultCard key={i} data={result} type={searchType} />)
                }
            </Flex>
        </Flex>
    );
}

export function ResultCard({ data, type }: { data: HFData, type: string }) {
    const { token } = theme.useToken();

    const onDragStart = useCallback((e: any) => {
        console.log(e);
        if (type === 'models') {
            bindDragData({ node: getHFPipelineNode(data.id) }, e);
        }
        if (type === 'datasets') {
            bindDragData({ node: getHFDatasetNode(data.id) }, e);
        }
    }, [data]);

    const onCopy = useCallback(() => {
        navigator.clipboard.writeText(data.id);
    }, [data]);

    const link = useMemo(() => {
        if (type === 'models') {
            return `${HF_HOST}/${data.id}`;
        } else {
            return `${HF_HOST}/datasets/${data.id}`;
        }
    }, [data, type]);

    return (
        <Card style={{ padding: 5, marginTop: 5, borderColor: '#eb9817' }} draggable onDragStart={onDragStart}>
            <Flex vertical style={{ marginBottom: 4 }}>
                <Flex justify='space-between'>
                    <Tooltip title={data.id} mouseEnterDelay={.5}>
                        <Flex style={{ overflow: 'hidden', whiteSpace: 'nowrap' }}>
                            <span style={{ display: 'inline-block', color: token.colorTextTertiary }}>{data.idParts[0]}/</span>
                            <span style={{ display: 'inline-block', textOverflow: 'ellipsis', overflow: 'hidden', fontWeight: 'bold' }}>{data.idParts[1]}</span>
                        </Flex>
                    </Tooltip>
                    <CopyOutlined onClick={onCopy} />
                </Flex>
                {
                    type === 'models' && <Text style={{ color: token.colorTextLabel }} ellipsis>{(data as HFModel).pipelineTag}</Text>
                }

                <Divider style={{ margin: '2px 0' }} />
            </Flex>
            {
                data.tags.slice(0, 10).map((tag: string, i: number) => (
                    <Tooltip title={tag} key={i} mouseEnterDelay={.5}>
                        <Tag style={tagStyle}>{tag}</Tag>
                    </Tooltip>
                ))
            }
            {
                data.tags.length > 10 &&
                <Tag style={{ ...tagStyle, maxWidth: '100px' }}>+{data.tags.length - 10} more</Tag>
            }
            <Flex justify='space-between'>
                <Link href={link} target="_blank">View {type === 'models' ? 'Model' : 'Dataset'}</Link>
                <Flex>
                    <div style={{ whiteSpace: 'nowrap' }}><HeartOutlined /> <TruncatedNumber number={data.likes} /></div>
                    &nbsp;
                    <div style={{ whiteSpace: 'nowrap' }}><DownloadOutlined /> <TruncatedNumber number={data.downloads} /></div>
                </Flex>
            </Flex>
        </Card>
    )
}

export function TruncatedNumber({ number }: { number: number }) {
    if (number >= 1000000000) {
        return <Text>{(number / 1000000000).toFixed(1)}B+</Text>;
    } else if (number >= 1000000) {
        return <Text>{(number / 1000000).toFixed(1)}M+</Text>;
    } else if (number >= 1000) {
        return <Text>{(number / 1000).toFixed(1)}K+</Text>;
    } else {
        return <Text>{number}</Text>;
    }
}


type GraphbookAPI = {
    useAPI: Function,
    useAPIMessage: Function,
    useAPINodeMessage: Function,
};

export function ExportPanels(graphbookAPI: GraphbookAPI) {
    return [{
        label: "Huggingface",
        children: <HF />,
        icon: "🤗"
    }];
}
