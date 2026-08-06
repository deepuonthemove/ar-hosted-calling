import { ChatDetail } from "@/components/chat-detail";

export default async function ChatPage({ params }: { params: Promise<{ accountUid: string }> }) {
  const { accountUid } = await params;
  return <ChatDetail accountUid={accountUid} />;
}
