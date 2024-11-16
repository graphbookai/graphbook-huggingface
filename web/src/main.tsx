import React, { useState, useCallback, useMemo, useEffect, useRef } from 'react';
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

export function HF({ useAPI }: { useAPI: Function }) {
    const [results, setResults] = useState<HFData[]>([]);
    const [searchTimeout, setSearchTimeout] = useState<NodeJS.Timeout>(null);
    const [searchType, setSearchType] = useState<string>('models');
    const [search, setSearch] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const schema = useRef<any>(null);
    const API = useAPI();

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

    useEffect(() => {
        const setSchema = async () => {
            const nodes = await API.getNodes();
            if (!nodes) {
                return;
            }
            console.log("Got nodes", nodes);
            const hfSteps = nodes.steps.Huggingface.children;
            schema.current = {
                datasets: hfSteps.HuggingfaceDataset,
                models: hfSteps.TransformersPipeline
            };
        };
        setSchema();
    }, [API]);

    return (
        <Flex vertical style={{ height: '100%', width: '100%' }}>
            <Search
                addonBefore={
                    <Select dropdownStyle={selectStyle} defaultValue="models" onChange={onSearchTypeChange}>
                        <Select.Option value="models">Models</Select.Option>
                        <Select.Option value="datasets">Datasets</Select.Option>
                    </Select>
                }
                style={{ marginBottom: 5 }} placeholder="Search" onChange={onSearchChange} loading={isLoading}
            />
            <Flex vertical style={{ overflowY: 'auto', paddingRight: 5 }}>
                {
                    results.map((result: HFData, i: number) => <ResultCard schema={schema.current[searchType]} key={i} data={result} type={searchType} />)
                }
            </Flex>
        </Flex>
    );
}

export function ResultCard({ data, type, schema }: { data: HFData, type: string, schema: any }) {
    const { token } = theme.useToken();

    const onDragStart = useCallback((e: any) => {
        const node = { type: "step", data: { label: schema.name, ...schema } };
        if (type === 'models') {
            node.data.parameters.model_id.value = data.id;
            bindDragData({ node }, e);
        }
        if (type === 'datasets') {
            node.data.parameters.dataset_id.value = data.id;
            bindDragData({ node }, e);
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
        label: "Hugging Face",
        children: <HF useAPI={graphbookAPI.useAPI} />,
        icon: "🤗"
    }];
}
