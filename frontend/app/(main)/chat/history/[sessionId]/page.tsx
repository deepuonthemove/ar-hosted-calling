import { ChatSessionDetail } from "@/components/chat-session-detail";

export default async function ChatSessionPage({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await params;
  return <ChatSessionDetail sessionId={sessionId} />;
}
