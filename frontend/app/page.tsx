import { Suspense } from "react";
import { Container, Typography } from "@mui/material";
import ClientHome from "./components/ClientHome";

export default function Home() {
  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Typography variant="h4" gutterBottom>Mentor–Mentee Matcher</Typography>
      <Typography variant="body1" gutterBottom>
        Matches mentees to mentors with <b>major first</b>, then extracurricular overlap.
      </Typography>

      <Suspense fallback={<div>Loading...</div>}>
        <ClientHome />
      </Suspense>
    </Container>
  );
}
