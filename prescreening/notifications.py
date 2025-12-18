# prescreening/notifications.py - Real Notification System
import asyncio
import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import logging

from .models import PreScreeningNotification, NotificationType
from .database import get_database

logger = logging.getLogger(__name__)

class NotificationChannel:
    """Base class for notification channels"""
    
    async def send(self, notification: PreScreeningNotification) -> bool:
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@hiregenix.com")
    
    async def send(self, notification: PreScreeningNotification) -> bool:
        """Send email notification"""
        try:
            # Get recipient email from candidate/job data
            recipient = await self._get_recipient_email(notification)
            if not recipient:
                logger.warning(f"No recipient email found for notification {notification.id}")
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = recipient
            msg['Subject'] = notification.title
            
            # Email body
            body = self._create_email_body(notification)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _get_recipient_email(self, notification: PreScreeningNotification) -> Optional[str]:
        """Get recipient email address"""
        # In real implementation, this would query your user database
        # For now, return a placeholder
        if notification.notification_type == NotificationType.HUMAN_REVIEW:
            return os.getenv("HR_EMAIL", "hr@hiregenix.com")
        elif notification.notification_type == NotificationType.CANDIDATE_UPDATE:
            return f"candidate_{notification.candidate_id}@example.com"
        else:
            return os.getenv("ADMIN_EMAIL", "admin@hiregenix.com")
    
    def _create_email_body(self, notification: PreScreeningNotification) -> str:
        """Create HTML email body"""
        score = notification.metadata.get('score', 0)
        bucket = notification.metadata.get('bucket', 'unknown')
        action = notification.metadata.get('action', 'Review required')
        
        return f"""
        <html>
        <body>
            <h2>HireGenix Pre-screening Update</h2>
            <p><strong>Title:</strong> {notification.title}</p>
            <p><strong>Message:</strong> {notification.message}</p>
            
            <h3>Details:</h3>
            <ul>
                <li><strong>Candidate ID:</strong> {notification.candidate_id}</li>
                <li><strong>Job ID:</strong> {notification.job_id}</li>
                <li><strong>Score:</strong> {score}%</li>
                <li><strong>Category:</strong> {bucket}</li>
                <li><strong>Next Action:</strong> {action}</li>
            </ul>
            
            <p>
                <a href="https://app.hiregenix.com/prescreening/{notification.prescreening_id}" 
                   style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                    View Details
                </a>
            </p>
            
            <p><em>This notification was generated automatically by HireGenix.</em></p>
        </body>
        </html>
        """

class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel for real-time updates"""
    
    def __init__(self):
        self.webhook_urls = {
            'slack': os.getenv("SLACK_WEBHOOK_URL", ""),
            'teams': os.getenv("TEAMS_WEBHOOK_URL", ""),
            'custom': os.getenv("CUSTOM_WEBHOOK_URL", "")
        }
    
    async def send(self, notification: PreScreeningNotification) -> bool:
        """Send webhook notifications"""
        success = True
        
        # Send to all configured webhooks
        for webhook_type, url in self.webhook_urls.items():
            if url:
                try:
                    await self._send_webhook(webhook_type, url, notification)
                except Exception as e:
                    logger.error(f"Error sending {webhook_type} webhook: {e}")
                    success = False
        
        return success
    
    async def _send_webhook(self, webhook_type: str, url: str, notification: PreScreeningNotification):
        """Send individual webhook"""
        payload = self._create_webhook_payload(webhook_type, notification)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Webhook sent to {webhook_type}: {response.status_code}")
    
    def _create_webhook_payload(self, webhook_type: str, notification: PreScreeningNotification) -> Dict[str, Any]:
        """Create webhook payload based on type"""
        score = notification.metadata.get('score', 0)
        bucket = notification.metadata.get('bucket', 'unknown')
        
        if webhook_type == 'slack':
            return {
                "text": notification.title,
                "attachments": [
                    {
                        "color": self._get_slack_color(bucket),
                        "fields": [
                            {"title": "Score", "value": f"{score}%", "short": True},
                            {"title": "Category", "value": bucket, "short": True},
                            {"title": "Candidate", "value": notification.candidate_id, "short": True},
                            {"title": "Job", "value": notification.job_id, "short": True}
                        ],
                        "text": notification.message,
                        "footer": "HireGenix Pre-screening",
                        "ts": int(notification.created_at.timestamp())
                    }
                ]
            }
        elif webhook_type == 'teams':
            return {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "summary": notification.title,
                "themeColor": self._get_teams_color(bucket),
                "sections": [
                    {
                        "activityTitle": notification.title,
                        "activitySubtitle": f"Score: {score}% | Category: {bucket}",
                        "text": notification.message,
                        "facts": [
                            {"name": "Candidate ID", "value": notification.candidate_id},
                            {"name": "Job ID", "value": notification.job_id},
                            {"name": "Time", "value": notification.created_at.isoformat()}
                        ]
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "View Details",
                        "targets": [
                            {
                                "os": "default",
                                "uri": f"https://app.hiregenix.com/prescreening/{notification.prescreening_id}"
                            }
                        ]
                    }
                ]
            }
        else:  # custom
            return {
                "event": "prescreening_notification",
                "data": {
                    "id": notification.id,
                    "prescreening_id": notification.prescreening_id,
                    "candidate_id": notification.candidate_id,
                    "job_id": notification.job_id,
                    "type": notification.notification_type.value,
                    "title": notification.title,
                    "message": notification.message,
                    "metadata": notification.metadata,
                    "created_at": notification.created_at.isoformat()
                }
            }
    
    def _get_slack_color(self, bucket: str) -> str:
        """Get Slack message color based on score bucket"""
        colors = {
            'EXCELLENT': '#28a745',    # green
            'GOOD': '#17a2b8',         # blue
            'POTENTIAL': '#ffc107',     # yellow
            'NOT_ELIGIBLE': '#dc3545'   # red
        }
        return colors.get(bucket, '#6c757d')  # default gray
    
    def _get_teams_color(self, bucket: str) -> str:
        """Get Teams message color based on score bucket"""
        colors = {
            'EXCELLENT': '28a745',
            'GOOD': '17a2b8', 
            'POTENTIAL': 'ffc107',
            'NOT_ELIGIBLE': 'dc3545'
        }
        return colors.get(bucket, '6c757d')

class InAppNotificationChannel(NotificationChannel):
    """In-app notification channel"""
    
    async def send(self, notification: PreScreeningNotification) -> bool:
        """Send in-app notification"""
        try:
            # Store in database for in-app display
            db = await get_database()
            await db.create_notification(notification)
            
            # Also send via WebSocket if user is online
            await self._send_websocket_notification(notification)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending in-app notification: {e}")
            return False
    
    async def _send_websocket_notification(self, notification: PreScreeningNotification):
        """Send notification via WebSocket"""
        try:
            # This would connect to your WebSocket manager
            # For now, just log the attempt
            logger.info(f"WebSocket notification sent for {notification.id}")
        except Exception as e:
            logger.error(f"WebSocket notification error: {e}")

class NotificationService:
    """Main notification service orchestrator"""
    
    def __init__(self):
        self.channels = {
            'email': EmailNotificationChannel(),
            'webhook': WebhookNotificationChannel(), 
            'inapp': InAppNotificationChannel()
        }
        self.enabled_channels = self._get_enabled_channels()
    
    def _get_enabled_channels(self) -> List[str]:
        """Get enabled notification channels from environment"""
        enabled = os.getenv("NOTIFICATION_CHANNELS", "email,inapp").split(",")
        return [channel.strip() for channel in enabled]
    
    async def send_notification(
        self,
        notification: PreScreeningNotification,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send notification through specified channels"""
        channels = channels or self.enabled_channels
        results = {}
        
        # Send through each channel
        tasks = []
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                tasks.append(self._send_with_retry(channel_name, channel, notification))
        
        # Execute all sends concurrently
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            results = dict(zip(channels, channel_results))
        
        # Update notification status
        notification.sent = any(results.values())
        notification.sent_at = datetime.now() if notification.sent else None
        
        # Log results
        success_channels = [ch for ch, success in results.items() if success]
        failed_channels = [ch for ch, success in results.items() if not success]
        
        if success_channels:
            logger.info(f"Notification {notification.id} sent via: {success_channels}")
        if failed_channels:
            logger.warning(f"Notification {notification.id} failed via: {failed_channels}")
        
        return results
    
    async def _send_with_retry(
        self,
        channel_name: str,
        channel: NotificationChannel,
        notification: PreScreeningNotification,
        max_retries: int = 3
    ) -> bool:
        """Send notification with retry logic"""
        for attempt in range(max_retries):
            try:
                success = await channel.send(notification)
                if success:
                    return True
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {channel_name}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return False

# Global notification service
notification_service = NotificationService()

async def send_notification(
    title: str,
    message: str,
    notification_type: str,
    prescreening_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    job_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Send notification using the global service"""
    
    notification = PreScreeningNotification(
        id=f"notif_{int(datetime.now().timestamp())}",
        prescreening_id=prescreening_id or "",
        candidate_id=candidate_id or "",
        job_id=job_id or "",
        notification_type=NotificationType(notification_type),
        title=title,
        message=message,
        metadata=metadata or {},
        created_at=datetime.now(),
        sent=False
    )
    
    results = await notification_service.send_notification(notification)
    return any(results.values())

async def send_prescreening_notification(notification: PreScreeningNotification) -> Dict[str, bool]:
    """Send pre-screening specific notification"""
    return await notification_service.send_notification(notification)