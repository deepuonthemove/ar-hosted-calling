import { ProjectView } from "@/components/project-view";

export default async function ProjectPage({ params }: { params: Promise<{ projectId: string }> }) {
  const { projectId } = await params;
  return <ProjectView projectId={projectId} />;
}
