"use client";
import { useState } from "react";
import Papa from "papaparse";
import axios from "axios";
import { Button, Typography, Stack, Paper, FormControl, InputLabel, Select, MenuItem, Switch, FormControlLabel, TextField } from "@mui/material";

type Mentor = { id: string; name?: string; major: string; extracurriculars: string[]; capacity: number; };
type Mentee = { id: string; name?: string; target_major: string; extracurriculars: string[]; };

type CsvRow = {
    id?: string;
    mentor_id?: string;
    mentee_id?: string;
    name?: string;
    major?: string;
    target_major?: string;
    extracurriculars?: string;
    capacity?: string | number;
};

type PriorityMode = "lexicographic" | "weighted";
type ExtrasMetric = "count" | "jaccard" | "cosine";

type Assignment = {
    mentee: string;
    mentee_name?: string;
    mentor: string;
    mentor_name?: string;
    major_match: boolean;
    extras_score: number;
};

type Results = {
    objective: {
        major_matches: number;
        extras_total: number;
    };
    assignments: Assignment[];
};

export default function ClientHome() {
    const [mentors, setMentors] = useState<Mentor[]>([]);
    const [mentees, setMentees] = useState<Mentee[]>([]);
    const [priorityMode, setPriorityMode] = useState<PriorityMode>("weighted");
    const [bigM, setBigM] = useState<number>(1000);
    const [extrasMetric, setExtrasMetric] = useState<ExtrasMetric>("count");
    const [results, setResults] = useState<Results | null>(null);
    const [backendUrl, setBackendUrl] = useState<string>(process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000");

    const parseCsv = (file: File, isMentor: boolean) => {
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (res) => {
                const rows = res.data as CsvRow[];
                if (isMentor) {
                    setMentors(rows.map(r => ({
                        id: r.id || r.mentor_id || "",
                        name: r.name,
                        major: r.major || "",
                        extracurriculars: (r.extracurriculars || "").split(/[;,]/).map((s: string) => s.trim()).filter(Boolean),
                        capacity: Number(r.capacity ?? 1)
                    })));
                } else {
                    setMentees(rows.map(r => ({
                        id: r.id || r.mentee_id || "",
                        name: r.name,
                        target_major: r.target_major || r.major || "",
                        extracurriculars: (r.extracurriculars || "").split(/[;,]/).map((s: string) => s.trim()).filter(Boolean)
                    })));
                }
            }
        });
    };

    const useSample = () => {
        setMentors([
            { id: "m1", name: "Alice", major: "Computer Science", extracurriculars: ["ACM", "Robotics"], capacity: 2 },
            { id: "m2", name: "Ben", major: "Data Science", extracurriculars: ["Math Club", "Hack Club"], capacity: 1 },
            { id: "m3", name: "Chitra", major: "Computer Science", extracurriculars: ["Robotics", "Volunteering"], capacity: 2 },
        ]);
        setMentees([
            { id: "s1", name: "Riya", target_major: "Computer Science", extracurriculars: ["Robotics", "Math Club"] },
            { id: "s2", name: "Evan", target_major: "Data Science", extracurriculars: ["ACM", "Hack Club"] },
            { id: "s3", name: "Maya", target_major: "Computer Science", extracurriculars: ["Volunteering"] },
            { id: "s4", name: "Leo", target_major: "Computer Science", extracurriculars: ["Robotics"] },
        ]);
    };

    const runSolve = async () => {
        const payload = {
            mentors,
            mentees,
            scoring: { priority_mode: priorityMode, bigM, extras_metric: extrasMetric, major_rule: "exact" }
        };
        const { data } = await axios.post<Results>(`${backendUrl}/solve`, payload);
        setResults(data);
    };

    return (
        <Stack spacing={3} sx={{ mt: 2 }}>
            <Paper sx={{ p: 2 }}>
                <Typography variant="h6">Data</Typography>
                <Stack direction="row" spacing={2} sx={{ mt: 1 }} alignItems="center">
                    <Button variant="outlined" component="label">Upload mentors.csv
                        <input hidden type="file" accept=".csv" onChange={(e) => e.target.files && parseCsv(e.target.files[0], true)} />
                    </Button>
                    <Button variant="outlined" component="label">Upload mentees.csv
                        <input hidden type="file" accept=".csv" onChange={(e) => e.target.files && parseCsv(e.target.files[0], false)} />
                    </Button>
                    <Button onClick={useSample}>Use sample data</Button>
                    <TextField size="small" label="Backend URL" value={backendUrl} onChange={e => setBackendUrl(e.target.value)} />
                </Stack>
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    mentors.csv: id,name,major,extracurriculars,capacity — mentees.csv: id,name,target_major,extracurriculars
                </Typography>
            </Paper>

            <Paper sx={{ p: 2 }}>
                <Typography variant="h6">Scoring & Priority</Typography>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
                    <FormControl size="small">
                        <InputLabel>Mode</InputLabel>
                        <Select value={priorityMode} label="Mode" onChange={(e) => setPriorityMode(e.target.value as PriorityMode)}>
                            <MenuItem value="weighted">Weighted</MenuItem>
                            <MenuItem value="lexicographic">Lexicographic</MenuItem>
                        </Select>
                    </FormControl>
                    <TextField size="small" type="number" label="bigM" value={bigM} onChange={e => setBigM(parseInt(e.target.value || "1000"))} disabled={priorityMode !== "weighted"} />
                    <FormControl size="small">
                        <InputLabel>Extras</InputLabel>
                        <Select value={extrasMetric} label="Extras" onChange={(e) => setExtrasMetric(e.target.value as ExtrasMetric)}>
                            <MenuItem value="count">Count</MenuItem>
                            <MenuItem value="jaccard">Jaccard</MenuItem>
                            <MenuItem value="cosine">Cosine</MenuItem>
                        </Select>
                    </FormControl>
                    <FormControlLabel control={<Switch defaultChecked />} label="One mentor per mentee (always on)" />
                </Stack>
            </Paper>

            <Stack direction="row" spacing={2}>
                <Button variant="contained" onClick={runSolve} disabled={!mentors.length || !mentees.length}>Run matching</Button>
                <Button onClick={() => setResults(null)}>Clear results</Button>
            </Stack>

            {results && (
                <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Results</Typography>
                    <Typography variant="body2">
                        Major matches: <b>{results.objective.major_matches}</b> • Extras total: <b>{results.objective.extras_total.toFixed(2)}</b>
                    </Typography>
                    <div style={{ marginTop: 12 }}>
                        {results.assignments.map((a: Assignment, idx: number) => (
                            <Paper key={idx} sx={{ p: 1, mb: 1 }}>
                                <Typography variant="body2">
                                    <b>{a.mentee_name || a.mentee}</b> → <b>{a.mentor_name || a.mentor}</b>
                                    {" "} | Major: {a.major_match ? "✓" : "✗"} | Extras: {a.extras_score.toFixed(2)}
                                </Typography>
                            </Paper>
                        ))}
                    </div>
                </Paper>
            )}
        </Stack>
    );
}
