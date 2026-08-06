import { ProjectRowView } from "@/components/project-row-view";

export default async function ProjectRowPage({ params }: { params: Promise<{ projectId: string; rowNum: string }> }) {
  const { projectId, rowNum } = await params;
  return <ProjectRowView projectId={projectId} rowNum={Number(rowNum)} />;
}
